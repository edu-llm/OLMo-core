"""Mirror a HuggingFace Trainer output directory to an S3 prefix.

WHY THIS FILE EXISTS. The eduLLM platform hands every container an
``EDULLM_CHECKPOINT_DIR`` that is an ``s3://`` URI, and the guide tells a workload to pass
it straight through as its save folder. OLMo-core's own Trainer understands that, because
``olmo_core.io`` dispatches on the scheme. This recipe does not use OLMo-core's Trainer, it
uses HuggingFace's, and ``transformers`` treats ``output_dir`` as a plain filesystem path.
``Trainer.save_model`` calls ``os.makedirs`` on the string it is given, so an ``s3://`` URI
produces a local directory literally named ``s3:`` with the rest of the URI nested inside
it. The container then exits and takes that directory with it.

That failure is silent in every direction a reviewer looks. Run
``run_019fc2e3-f632-7062-a8b4-e67132f53c0c`` exited 0, trained 32 steps with a moving loss,
logged to W&B under the right project, printed ``Saved model + tokenizer to s3://...`` and
left the S3 prefix empty. The print was the script echoing its own ``output_dir`` back, not
a confirmation that anything had been uploaded. Everything below exists so that the print
is only reachable after the bytes are in the bucket and listable.

WHY boto3 DIRECTLY RATHER THAN olmo_core.io. ``olmo_core.io`` already has upload and
download helpers for exactly this, and reaching for them would be the smaller diff. It is
the wrong dependency to take. The whole p7/impl3 tree imports nothing from ``olmo_core``,
which is what lets a researcher run it in a plain conda environment on ORCD where no
``olmo_core`` is installed, and what keeps the recipe from inheriting the pinned torch that
package needs. boto3 is already in the image, is already a hard requirement of the S3 path
in ``olmo_core.io`` itself, and adds no new install anywhere.

WHY THE STAGING DIRECTORY IS A REAL LOCAL DIRECTORY RATHER THAN AN S3 FILESYSTEM SHIM.
The alternative was a fsspec or s3fs layer that makes ``output_dir`` look writable in
place. That fails on the parts of the checkpoint HF writes with something other than plain
``open``. ``safetensors`` memory-maps, ``torch.save`` seeks, and the atomic-rename dance HF
does around ``trainer_state.json`` has no meaning in an object store. Training to local
disk and copying afterwards is the boring option and is the one that cannot be broken by a
library deciding to seek.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

S3_SCHEME = "s3://"

# Where a run stages its checkpoints when output_dir is an S3 URI. Overridable because a
# Batch host's /tmp is not always the roomiest volume it has, and a full run of this recipe
# writes twelve LoRA checkpoints of roughly 140 MB each.
STAGING_DIR_ENV = "EDULLM_LOCAL_OUTPUT_DIR"

# The object this module writes before training starts. See ``announce_start``.
START_MARKER_NAME = "_run_metadata.json"

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)(?:/|$)")


def is_s3_uri(path) -> bool:
    return isinstance(path, str) and path.startswith(S3_SCHEME)


def split_s3_uri(uri: str):
    """``s3://bucket/a/b/`` -> ``("bucket", "a/b")``. Trailing slashes are dropped."""
    bucket, _, key = uri[len(S3_SCHEME):].partition("/")
    key = key.strip("/")
    if not bucket:
        raise ValueError(f"not a usable S3 URI, no bucket in {uri!r}")
    if not key:
        # Refused rather than defaulted. A bucket root is never what anyone means by a
        # checkpoint directory, and the workload role only grants writes under
        # teams/*/runs/*, so a run that got here would fail later with AccessDenied and a
        # message about permissions rather than about the empty path that caused it.
        raise ValueError(f"S3 URI names a bucket root with no prefix, refusing to use {uri!r}")
    return bucket, key


def staging_dir_for(uri: str) -> str:
    override = os.environ.get(STAGING_DIR_ENV)
    if override:
        return override
    _, key = split_s3_uri(uri)
    # Derived from the key rather than random, so two invocations pointed at the same
    # prefix reuse one directory and the second finds the first's work instead of
    # re-downloading it. The key carries the run id, which is a uuid, so distinct runs
    # cannot collide.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")
    return os.path.join(tempfile.gettempdir(), "edullm-sft", slug)


def _human(n_bytes: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n_bytes < 1024 or unit == "GiB":
            return f"{n_bytes:.0f} {unit}" if unit == "B" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} GiB"


class S3Mirror:
    """One-way copy of a local training directory into one S3 prefix.

    One way on purpose. The mirror never deletes a remote object, including when
    ``save_total_limit`` makes HF rotate a local checkpoint away. The divergence that
    creates is the safe direction, S3 keeps more than local disk does, and the resume path
    below only ever asks for the newest step so a surplus of older ones costs nothing but
    storage. Mirroring the deletion would mean a bug in this file could remove the only
    copy of a checkpoint the run cannot reproduce, which is a much worse failure than
    paying for a few adapter files nobody reads.
    """

    def __init__(self, uri: str, local_dir: str):
        self.uri = uri.rstrip("/") + "/"
        self.bucket, self.prefix = split_s3_uri(uri)
        self.local_dir = local_dir
        self._client = None
        # relative path -> (size, mtime_ns) of what has already been sent. Fingerprinted
        # rather than merely named so that the final save_model, which rewrites
        # adapter_model.safetensors at the root over whatever a mid-run save left there,
        # is recognised as a new object and sent again.
        self._sent: dict[str, tuple[int, int]] = {}
        self._sent_keys: set[str] = set()

    @classmethod
    def for_output_dir(cls, output_dir: str):
        """The mirror for an ``s3://`` output dir, or ``None`` for a local one.

        Returning None rather than a no-op mirror keeps the ordinary local path free of
        this file entirely, which matters because that is the path every ORCD run and every
        developer takes and none of them should acquire a boto3 import.
        """
        if not is_s3_uri(output_dir):
            return None
        return cls(output_dir, staging_dir_for(output_dir))

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            # Retries are the reason this is not a bare ``boto3.client("s3")``. A twelve
            # hour run makes hundreds of PUTs and S3 answers a few of them with a 500 or a
            # SlowDown as a matter of course. The default legacy mode retries five times
            # on a narrower set of errors; standard mode covers throttling and the
            # transient 5xx family, which is exactly the set that must not end a training
            # run. What is deliberately NOT here is a fallback that gives up and continues,
            # see ``upload_pending``.
            self._client = boto3.client(
                "s3", config=Config(retries={"max_attempts": 8, "mode": "standard"})
            )
        return self._client

    def key_for(self, relative_path: str) -> str:
        return f"{self.prefix}/{relative_path.replace(os.sep, '/')}"

    def announce_start(self, metadata: dict) -> str:
        """Write the run's metadata object, which doubles as the write-capability probe.

        THIS IS THE FAIL-FAST AND IT IS THE HALF OF THE FIX THAT SAVES MONEY. Everything
        else here runs at the first checkpoint, which on a real run is an hour in and on a
        twelve hour run can be later than that. A prefix this container cannot write to is
        knowable in the first two seconds, and finding out then costs a startup instead of
        an A10G hour. The permission that matters is PutObject, which listing does not
        exercise, so the probe has to be a write.

        It is a real artifact rather than a throwaway probe because the workload role's
        DeleteObject grant is narrower than its PutObject grant, so a probe this code
        wanted to clean up afterwards would be a second permission to depend on for no
        gain. Recording which run id, commit and base model wrote a prefix is worth having
        anyway, and it is the first thing an ``aws s3 ls`` of a live run will show.
        """
        key = f"{self.prefix}/{START_MARKER_NAME}"
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        self._sent_keys.add(key)
        print(f"s3: write access confirmed, wrote s3://{self.bucket}/{key}")
        return key

    def _local_files(self):
        for root, _dirs, files in os.walk(self.local_dir):
            for name in files:
                yield os.path.join(root, name)

    def upload_pending(self, label: str):
        """Send every local file whose size or mtime differs from what was last sent.

        NO try/except ANYWHERE IN THIS METHOD, AND THAT IS THE DESIGN RATHER THAN AN
        OVERSIGHT. A caught upload error would be logged, the run would carry on, the
        process would exit 0, and the prefix would be empty. That is precisely the defect
        this file was written to remove, reintroduced one level up with a warning line in a
        log nobody reads instead of no line at all. boto3 has already retried the transient
        cases eight times by the time an exception reaches here, so an exception at this
        point means the destination is genuinely unwritable, and a training run that cannot
        write its output has nothing left worth spending a GPU hour on.
        """
        sent_files = 0
        sent_bytes = 0
        for path in sorted(self._local_files()):
            relative = os.path.relpath(path, self.local_dir)
            stat = os.stat(path)
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            if self._sent.get(relative) == fingerprint:
                continue
            key = self.key_for(relative)
            self.client.upload_file(path, self.bucket, key)
            self._sent[relative] = fingerprint
            self._sent_keys.add(key)
            sent_files += 1
            sent_bytes += stat.st_size
        if sent_files:
            print(f"s3: uploaded {sent_files} file(s), {_human(sent_bytes)} "
                  f"[{label}] -> {self.uri}")
        else:
            print(f"s3: nothing new to upload [{label}]")
        return sent_files, sent_bytes

    def list_remote(self) -> dict[str, int]:
        """Every object under the prefix, as key -> size."""
        found: dict[str, int] = {}
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": f"{self.prefix}/"}
            if token:
                kw["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kw)
            for obj in page.get("Contents", ()):
                found[obj["Key"]] = obj["Size"]
            if not page.get("IsTruncated"):
                return found
            token = page["NextContinuationToken"]

    def verify(self):
        """Read the prefix back and confirm everything this process sent is in it.

        A successful PUT is already an acknowledgement from S3 and this listing should
        never disagree with it. It is here because the bug being fixed was a run that
        believed its own report, and a check that costs one LIST at the end of a training
        run is the cheapest possible way to make the success message mean something. It
        also catches the case no PUT can, which is a run pointed at a prefix that some
        other process is concurrently clearing.
        """
        remote = self.list_remote()
        missing = sorted(self._sent_keys - set(remote))
        if missing:
            raise RuntimeError(
                f"S3 verification failed. {len(missing)} object(s) uploaded to "
                f"{self.uri} are not in a listing of it, first few: {missing[:5]}"
            )
        if not remote:
            raise RuntimeError(
                f"S3 verification failed. Nothing is under {self.uri} after training. "
                "The checkpoints did not survive the run."
            )
        total = sum(remote.values())
        print(f"s3: verified {len(remote)} object(s), {_human(total)} under {self.uri}")
        return len(remote), total

    def latest_remote_checkpoint_step(self):
        steps = set()
        for key in self.list_remote():
            relative = key[len(self.prefix) + 1:]
            match = _CHECKPOINT_DIR.match(relative)
            if match:
                steps.add(int(match.group(1)))
        return max(steps) if steps else None

    def hydrate_latest_checkpoint(self):
        """Download the newest remote ``checkpoint-N`` into the staging dir.

        THIS IS WHAT MAKES A SECOND ATTEMPT MEAN ANYTHING. The training profiles set
        ``maximum_attempts: 2`` with ``resume_required: true``, and Batch re-runs a retry
        as the same container with the same environment and therefore the same
        ``EDULLM_CHECKPOINT_DIR``. Attempt 2 starts on a new host with an empty /tmp, so
        without this it finds no local checkpoint, restarts from step 0, and the second
        attempt is the first one again at full price. That is the failure the workload
        catalog's comment on this profile warns about.

        ONLY THE NEWEST STEP, WHICH IS A DELIBERATE ASYMMETRY WITH THE UPLOAD. Every
        checkpoint goes up because the KL-forgetting curve is computed over all of them.
        Only one comes back down because resuming needs exactly one, and pulling twelve
        would add minutes of transfer and roughly 1.7 GB to a host that is about to write
        its own. The older checkpoints stay in S3 where the eval stage reads them from.
        """
        step = self.latest_remote_checkpoint_step()
        if step is None:
            print(f"s3: no checkpoint under {self.uri} to resume from")
            return None
        wanted = f"{self.prefix}/checkpoint-{step}/"
        local_checkpoint = os.path.join(self.local_dir, f"checkpoint-{step}")
        n_files = 0
        n_bytes = 0
        for key, size in sorted(self.list_remote().items()):
            if not key.startswith(wanted):
                continue
            destination = os.path.join(self.local_dir, key[len(self.prefix) + 1:])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            self.client.download_file(self.bucket, key, destination)
            # Recorded as already sent so the first save of the resumed run does not
            # re-upload a checkpoint that is the reason this run could start at all.
            stat = os.stat(destination)
            self._sent[os.path.relpath(destination, self.local_dir)] = (
                stat.st_size, stat.st_mtime_ns,
            )
            self._sent_keys.add(key)
            n_files += 1
            n_bytes += size
        print(f"s3: hydrated checkpoint-{step} ({n_files} files, {_human(n_bytes)}) "
              f"into {local_checkpoint}")
        return local_checkpoint

    def callback(self):
        """A ``TrainerCallback`` that uploads after every checkpoint HF writes.

        PER CHECKPOINT RATHER THAN ON COMPLETION, WHICH IS THE MORE EXPENSIVE OF THE TWO
        AND THE ONLY ONE THAT MATCHES WHAT THE PLATFORM PROMISES. The training profiles
        declare a checkpoint contract with ``interval_minutes: 30`` and
        ``resume_required: true``, which is a statement that a lost host costs at most half
        an hour. Uploading only at the end makes that statement false. A twelve hour run
        that loses its spot instance at hour eleven would have nothing in the bucket, and
        the retry the contract provides for would have nothing to resume from.

        WHAT IT COSTS, MEASURED RATHER THAN WAVED AT. ``save_total_limit`` is null in both
        configs and ``min_checkpoints`` is 10, and a full run of this recipe writes twelve
        checkpoints. A LoRA r=16 checkpoint of a 1B model is about 46 MB of adapter plus
        its optimizer and scheduler state, so the whole run pushes on the order of 1.7 GB
        in around 120 PUTs. At S3 list prices that is fractions of a cent in requests and
        about four cents a month to store, against $5.67 an hour for the GPU it protects.
        The cadence is not free, it is just enormously cheaper than the thing it insures.

        ON on_save RATHER THAN A TIMER. HF calls ``on_save`` after ``_save_checkpoint`` has
        returned, so the directory is complete when this runs and there is no window in
        which a half-written safetensors file is uploaded. A wall-clock timer would have to
        invent that guarantee for itself.
        """
        from transformers import TrainerCallback

        mirror = self

        class _S3UploadCallback(TrainerCallback):
            def on_save(self, args, state, control, **kw):
                # Rank zero only. Every rank runs the callback handler, but only the
                # saving process wrote anything, and letting eight ranks upload the same
                # adapter concurrently would multiply the traffic by eight and race on
                # identical keys for no benefit.
                if not state.is_world_process_zero:
                    return control
                mirror.upload_pending(label=f"step {state.global_step}")
                return control

        return _S3UploadCallback()
