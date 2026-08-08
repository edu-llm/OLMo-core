import hashlib
import io
import tarfile
import tomllib
from pathlib import Path

from olmo_core.hpo import artifacts
from olmo_core.hpo.artifacts import ensure_ftpfn_artifact


def test_pinned_ftpfn_and_ifbo_constants():
    # The FT-PFN weight artifact and ifBO package are pinned exactly so that a run is
    # reproducible and the downloaded artifact can be checksum-verified before a trusted load.
    assert artifacts.IFBO_PACKAGE_VERSION == "0.4.1"
    assert artifacts.IFBO_COMMIT == "8ddcef0ed1ca88f2992108d39876e926aa58b0f2"
    assert artifacts.FTPFN_MODEL_VERSION == "0.0.1"
    assert artifacts.FTPFN_ARCHIVE_URL == "https://api.figshare.com/v2/file/download/61709839"
    assert artifacts.FTPFN_ARTIFACT_FILENAME == "bopfn_broken_unisep_1000curves_10params_2M.pt"
    assert artifacts.FTPFN_ARCHIVE_MD5 == "eb7567eaae91f2a958bf81083655f97b"
    assert (
        artifacts.FTPFN_ARCHIVE_SHA256
        == "989bc724e832b272f2608c0204cc0ed4f2728dfa835a2525b5eed275236c12d4"
    )
    assert artifacts.FTPFN_ARTIFACT_MD5 == "d857292ca08c31fa18805e66e83e3437"
    assert (
        artifacts.FTPFN_ARTIFACT_SHA256
        == "2626a7955f6c607008e979dcf8bf4cd524c0b6dc696de7e415f58d616c814c69"
    )


def test_ensure_ftpfn_artifact_downloads_verifies_extracts_and_caches(monkeypatch, tmp_path):
    checkpoint_bytes = b"public ftpfn checkpoint"
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        member = tarfile.TarInfo(f"nested/{artifacts.FTPFN_ARTIFACT_FILENAME}")
        member.size = len(checkpoint_bytes)
        archive.addfile(member, io.BytesIO(checkpoint_bytes))

    archive_bytes = source_archive.read_bytes()
    monkeypatch.setattr(
        artifacts,
        "FTPFN_ARCHIVE_MD5",
        hashlib.md5(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        artifacts,
        "FTPFN_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        artifacts,
        "FTPFN_ARTIFACT_MD5",
        hashlib.md5(checkpoint_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        artifacts,
        "FTPFN_ARTIFACT_SHA256",
        hashlib.sha256(checkpoint_bytes).hexdigest(),
    )
    monkeypatch.setattr(artifacts, "FTPFN_ARCHIVE_URL", source_archive.as_uri())

    cache = tmp_path / "cache"
    artifact_path = ensure_ftpfn_artifact(cache)
    assert artifact_path.read_bytes() == checkpoint_bytes

    monkeypatch.setattr(
        artifacts,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit must not download")
        ),
    )
    assert ensure_ftpfn_artifact(cache) == artifact_path


def test_license_provenance_recorded():
    # ifBO code is MIT; the FT-PFN weights are CC BY 4.0 and require attribution + a
    # modification note, so those facts must live in-repo, not just in the plan.
    assert artifacts.IFBO_CODE_LICENSE == "MIT"
    assert artifacts.FTPFN_ARTIFACT_LICENSE == "CC-BY-4.0"
    assert artifacts.FTPFN_ARTIFACT_SOURCE.endswith("/31286173")


def test_max_hp_dimensions_matches_ftpfn_contract():
    # FT-PFN v0.0.1 supports at most 10 non-fidelity hyperparameter dimensions.
    assert artifacts.FTPFN_MAX_HP_DIMS == 10


def _repo_root() -> Path:
    # src/test/hpo/artifacts_test.py -> repo root is three parents up from src/test.
    return Path(__file__).resolve().parents[3]


def test_pyproject_declares_hpo_optional_dependency_group():
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "hpo" in extras, "expected an 'hpo' optional-dependencies group"
    joined = " ".join(extras["hpo"])
    assert "ifbo" in joined.lower()
    assert "unit-scaling==0.3.5" in joined.lower()
    assert "cmaes" not in joined.lower()
    assert "cmaes" in " ".join(extras["hpo-cma"]).lower()


def test_edullm_image_installs_hpo_and_bakes_verified_ftpfn():
    dockerfile = (_repo_root() / ".edullm" / "Dockerfile").read_text()
    assert "('wandb', 'hpo')" in dockerfile
    assert '".[wandb,hpo]"' in dockerfile
    assert "OLMO_CORE_HPO_ARTIFACT_CACHE=/opt/olmo-artifacts" in dockerfile
    assert "ensure_ftpfn_artifact" in dockerfile
    assert "import ifbo, unit_scaling" in dockerfile
