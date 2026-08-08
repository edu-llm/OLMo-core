import hashlib
import json
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_WORKSPACE = "/home/vs/AlphaAI/" + "memorysplit-requery-exact"
INVENTORY_PATH = ARCHIVE_ROOT / "archive-inventory.json"


def test_archived_sources_do_not_depend_on_private_workspace() -> None:
    offenders = []
    for subtree in ("scripts", "tests"):
        for path in (ARCHIVE_ROOT / subtree).rglob("*.py"):
            if PRIVATE_WORKSPACE in path.read_text(encoding="utf-8", errors="replace"):
                offenders.append(str(path.relative_to(ARCHIVE_ROOT)))

    assert offenders == []


def test_archive_inventory_matches_tracked_snapshot() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    actual = {}
    for path in ARCHIVE_ROOT.rglob("*"):
        if (
            path.is_file()
            and path != INVENTORY_PATH
            and "__pycache__" not in path.parts
        ):
            actual[str(path.relative_to(ARCHIVE_ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    assert inventory["schema"] == "p3-git-archive-inventory/v1"
    assert inventory["files"] == actual
