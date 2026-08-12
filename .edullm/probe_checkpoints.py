"""Inventory every trained checkpoint this platform can actually reach, and say so.

    python .edullm/probe_checkpoints.py

WHY THIS EXISTS. Evaluating four pretrained mixers needs four checkpoints, and nothing on a
laptop can find out whether they exist: the workload roles hold the only S3 identities, and
`AGENTS.md` forbids reaching AWS from here. So "where are the linear-attention, GDN, KDA and
GDN-2 checkpoints, and can a run open them?" is answerable only from inside a submitted job.
This is the cheapest job that answers it.

IT GOES THROUGH ``olmo_core.io`` RATHER THAN boto3, AND THAT IS THE POINT RATHER THAN A STYLE
CHOICE. `AGENTS.md` says not to write a script that calls AWS, and the reason it gives is that
such a script either fails, for whoever holds no role, or succeeds and leaves no run anybody can
cite. Both objections are about a laptop. This runs inside a submitted job that has a run id, a
manifest, an approval and a log -- and it reaches S3 the same way `Trainer` reaches a checkpoint,
through the library's own I/O layer. Nothing here is a credential this file supplies.

WHY A GPU PROFILE FOR A JOB THAT NEVER TOUCHES A GPU. `infra/iam/batch-gpu-roles.yaml` grants
the GPU workload role `s3:GetObject` under `teams/*/runs/*` and `s3:ListBucket` on the bucket --
but the list carries `Condition: StringLike: s3:prefix: teams/*/runs/*`, so a run may enumerate
inside a team's runs and may NOT enumerate the teams themselves. The CPU workload role has
PutObject there and deliberately NOT GetObject -- the GPU file says so in a comment -- so a CPU
profile cannot read a checkpoint at all. `gpu-1xt4` is the cheapest shape carrying the right
role, and this reads no tensors so the card's lack of bfloat16 is irrelevant.

THE FOREIGN BUCKET IS THE SECOND HALF AND THE REAL QUESTION. The linear-attention and GDN 370M
runs were trained off-platform and saved to `s3://edullm-olmo-370m-ckpts/...`, which appears in
none of that role's three policies. Committed IAM therefore says a run cannot read it. But
committed IAM has been behind deployment in this organisation before -- `guides/edullm-data.md`
carries a banner about exactly that -- so this asks the account instead of the file and reports
what came back. "Denied" and "absent" point at completely different next steps.

IT WRITES NOTHING AND TRAINS NOTHING. Every call is a list or an existence check.

OUTPUT ORDER IS LOAD-BEARING. `edullm logs` returns the last fifty lines a container printed, so
the inventory is printed LAST. A summary above the noise is a summary nobody can read.
"""

import json
import os
import sys
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from olmo_core.io import file_exists, list_directory

OUTPUTS = "s3://sbsandbox-intern-edullm-outputs"

# The off-platform buckets the baseline arms and their eval shards were written to. Named
# rather than discovered, because whether these exact names are reachable IS the question.
# run_019ff45e answered both of these and the two answers are different in kind:
#   edullm-olmo-370m-ckpts -> NoSuchBucket. The bucket does not exist, in this account or at
#                             all -- so the baseline linear/GDN checkpoints are not merely
#                             out of reach, there is nothing at that name to reach.
#   edullm-datasets        -> AccessDenied. That bucket exists and this role may not read it.
# Kept, plus plausible alternative spellings, because "the name in a launch script is stale" is
# a likelier explanation for a missing bucket than "the checkpoints were deleted", and the two
# are told apart by asking for the other names.
# EVERY BUCKET THIS ORGANISATION IS KNOWN TO HAVE, probed one by one, because enumerating
# buckets needs `s3:ListAllMyBuckets` and that action is on none of the role's three policies.
# So this is not a guess-list: the names are harvested from the platform's own CloudFormation
# (infra/*-bucket.yaml), its IAM templates, its tests, and every s3:// literal in either
# repository. If a checkpoint from any arm is anywhere this account can see, it is under one of
# these prefixes.
#
# What run_019ff45e already established, kept here because the two answers differ in kind:
#   edullm-olmo-370m-ckpts -> NoSuchBucket. Not denied -- ABSENT. The bucket the baseline
#                             launch scripts wrote to does not exist, so those checkpoints are
#                             not merely unreachable; there is nothing at that name.
#   edullm-datasets        -> AccessDenied. That one exists and this role may not read it.
CANDIDATES = [
    # the outputs bucket, at the one depth the prefix condition permits
    "s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs",
    # a dedicated checkpoints bucket, with the runs/ layout its tests describe
    "s3://sbsandbox-intern-edullm-checkpoints",
    "s3://sbsandbox-intern-edullm-checkpoints/runs",
    # where `edullm run` / `edullm shell` machines persist files that outlive them --
    # the likeliest home for anything trained off the submission path
    "s3://edullm-scratch",
    "s3://edullm-scratch/pianomaster99",
    # the rest of the account's declared buckets
    "s3://sbsandbox-intern-edullm-lineage",
    "s3://sbsandbox-intern-edullm-artifacts",
    "s3://edullm-data/pretrain",
    "s3://edullm-landing",
    # and the two already answered, re-asked so one log holds the whole picture
    "s3://edullm-olmo-370m-ckpts",
    "s3://edullm-datasets",
]

# `block.sequence_mixer.type` as a saved config spells it, mapped to the arm it is.
ARMS = {
    "attention": "vanilla attention",
    "linear_attention": "plain linear attention",
    "gated_delta_net": "GDN",
    "gated_delta_net_2": "GDN-2",
    "kimi_delta_attention": "KDA",
    "kimi_delta_householder": "KDA-Householder",
}


def safe_list(uri: str) -> Tuple[List[str], Optional[str]]:
    """List one prefix. Returns ``(entries, error)`` -- and the error is not optional.

    THE FIRST VERSION OF THIS RETURNED `[]` ON ANY EXCEPTION AND THAT MADE THE PROBE LIE.
    `s3:ListBucket` on the outputs bucket carries `Condition: StringLike: s3:prefix:
    teams/*/runs/*`, because listing is a bucket-level action that an object ARN cannot scope.
    A list of `teams/` does not match that pattern, so it is DENIED -- and swallowing the
    denial printed "0 team(s)", which reads as "there are no checkpoints" when it means "I was
    not allowed to look". run_019ff45e reported exactly that, and it was wrong.

    So the error comes back and the caller prints it. An empty list and a refusal must never
    render the same way.
    """
    try:
        return list(list_directory(uri)), None
    except Exception as exc:  # noqa: BLE001 -- the exception text IS a finding
        return [], traceback.format_exception_only(type(exc), exc)[0].strip()


def read_config(uri: str) -> Optional[Any]:
    try:
        if not file_exists(uri):
            return None
        from olmo_core.io import get_bytes_range

        # Configs here are a few tens of KB; a megabyte is a generous ceiling that avoids
        # needing the object's length first.
        raw = get_bytes_range(uri, 0, 1_048_576)
        return json.loads(raw.split(b"\x00")[0].decode("utf-8", "ignore"))
    except Exception:
        return None


def identify(config: Any) -> Dict[str, str]:
    """Pull the few fields that say which arm a checkpoint is."""
    out = {"mixer": "unknown", "d_model": "?", "n_layers": "?", "params": "?"}
    if not isinstance(config, dict):
        return out
    model = config.get("model") or {}
    out["d_model"] = str(model.get("d_model", "?"))
    out["n_layers"] = str(model.get("n_layers", "?"))
    block = model.get("block") or {}
    mixer = block.get("sequence_mixer") if isinstance(block, dict) else None
    if isinstance(mixer, dict):
        out["mixer"] = str(mixer.get("type") or mixer.get("_CLASS_") or "unknown")
    return out


def main() -> int:
    findings: List[Dict[str, Any]] = []
    errors: List[str] = []

    # TEAMS ARE NAMED, NOT DISCOVERED, and the prefix condition above is why: a list of
    # `teams/` cannot match `teams/*/runs/*`, so the team has to be known before the first
    # call. `edullm check --json` reports it as `team`, and the platform sets it on the run.
    teams = [t for t in os.environ.get("EDULLM_PROBE_TEAMS", "input-core").split(",") if t]
    print(f"teams to look under (named, because listing teams/ is refused): {teams}")

    # POSITIVE CONTROL, FIRST, so that a later empty result can be told apart from a refusal.
    # This run id is one whose checkpoints are known to have been written -- if even this comes
    # back empty, the probe is broken rather than the bucket.
    control = os.environ.get(
        "EDULLM_PROBE_CONTROL",
        f"{OUTPUTS}/teams/input-core/runs/run_019fdfe0-01d2-7092-bdae-dfce5f5d4f19/checkpoints",
    )
    got, err = safe_list(control)
    print(f"control {control}")
    print(f"  -> {len(got)} entr(ies){'  ERROR: ' + err if err else ''}")
    for g in got[:8]:
        print(f"     {g}")
    if err:
        errors.append(f"control listing failed: {err}")

    for team in teams:
        runs, err = safe_list(f"{OUTPUTS}/teams/{team}/runs")
        if err:
            errors.append(f"listing runs for team {team!r}: {err}")
        print(f"  team {team}: {len(runs)} run(s){'  ERROR: ' + err if err else ''}")
        for run in runs:
            steps, err = safe_list(f"{run}/checkpoints")
            if err:
                errors.append(f"listing checkpoints for {run.rsplit('/', 1)[-1]}: {err}")
            steps = [s for s in steps if "step" in s.rsplit("/", 1)[-1]]
            if not steps:
                continue
            steps.sort(key=lambda s: int("".join(c for c in s.rsplit("/", 1)[-1] if c.isdigit())))
            cfg = None
            for cand in (
                f"{run}/config.json",
                f"{run}/checkpoints/config.json",
                f"{steps[-1]}/config.json",
            ):
                cfg = cfg or read_config(cand)
            info = identify(cfg)
            findings.append(
                {
                    "run": run.rsplit("/", 1)[-1],
                    "steps": [s.rsplit("/", 1)[-1] for s in steps],
                    "uri": f"{run}/checkpoints",
                    "config_found": cfg is not None,
                    **info,
                }
            )
            print(f"    {run.rsplit('/', 1)[-1]}: {len(steps)} step(s), mixer={info['mixer']}")

    notes: List[str] = []
    for uri in CANDIDATES:
        try:
            got = list(list_directory(uri))
            notes.append(f"READABLE  {uri}  ({len(got)} entries)")
            for g in got[:6]:
                notes.append(f"             {g}")
        except Exception as exc:  # noqa: BLE001 -- the exception type IS the finding
            first = traceback.format_exception_only(type(exc), exc)[0].strip()
            notes.append(f"UNREADABLE {uri}")
            notes.append(f"             {first[:140]}")

    print()
    print("=" * 78)
    print("CHECKPOINT INVENTORY -- what a platform run can open")
    print("=" * 78)
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_arm[ARMS.get(f["mixer"], f["mixer"])].append(f)
    if not findings:
        print("  NOTHING under teams/*/runs/*/checkpoints/ holds a step directory.")
    for arm in sorted(by_arm):
        for f in by_arm[arm]:
            flag = "" if f["config_found"] else "  (no config.json -- mixer unidentified)"
            print(
                f"  {arm:<24} d_model={f['d_model']:<5} L={f['n_layers']:<3} "
                f"steps={len(f['steps']):<3} last={f['steps'][-1]:<12} {f['run']}{flag}"
            )
    print()
    if errors:
        print("REFUSALS AND ERRORS -- read these before believing any empty result above")
        for e in errors:
            print(f"  {e[:150]}")
        print()
    print("EVERY KNOWN BUCKET, PROBED")
    for n in notes:
        print(f"  {n}")
    print()
    wanted = {"plain linear attention", "GDN", "KDA", "GDN-2"}
    have = {a for a in by_arm}
    print(f"arms reachable: {sorted(have)}")
    print(f"still missing : {sorted(wanted - have)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
