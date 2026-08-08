import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[3]
ENTRYPOINT = ROOT / ".edullm/model_arch_tests.py"
TRAIN_RUNNER = ROOT / ".edullm/train_core6_arm.py"
RUN_SPEC = ROOT / ".edullm/run-comparison.yaml"
SEEDS = ROOT / "docs/mamba-comparison/seeds.json"
DOCKERFILE = ROOT / ".edullm/Dockerfile"


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return values


def test_entrypoint_freezes_four_full_architecture_arms():
    values = _literal_assignments(ENTRYPOINT)
    assert values["ARMS"] == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd")
    assert values["ATTENTION_LAYERS"] == (3, 7, 11, 15)
    assert values["FROZEN_STEPS"] == 3721
    assert values["FROZEN_GLOBAL_BATCH_SIZE"] == 524288


def test_arm_major_five_seed_wave_is_machine_readable():
    assert RUN_SPEC.is_file()
    assert SEEDS.is_file()
    schedule = json.loads(SEEDS.read_text())
    expected_arms = ["mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd"]
    assert schedule["arms"] == expected_arms
    assert schedule["replicates_per_arm"] == 5
    assert schedule["fanout_size"] == 20
    assert schedule["cell_order"] == [arm for arm in expected_arms for _ in range(5)]
    assert schedule["steps"] == 3721
    assert schedule["global_batch_size"] == 524288
    assert schedule["tokens_per_cell"] == 1_950_875_648
    assert schedule["target_tokens_per_parameter"] == 5.0

    run_yaml = RUN_SPEC.read_text()
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in run_yaml
    assert "--steps 3721" in run_yaml
    assert "--global-batch-size 524288" in run_yaml

    def shell_array(name: str) -> list[str]:
        match = re.search(rf"{name}=\((.*?)\) &&", run_yaml, flags=re.DOTALL)
        assert match is not None
        return match.group(1).split()

    assert shell_array("ARMS") == schedule["cell_order"]
    assert shell_array("DSEEDS") == [
        str(seed) for _ in expected_arms for seed in schedule["data_seeds"]
    ]
    assert shell_array("ISEEDS") == [
        str(seed) for arm in expected_arms for seed in schedule["init_seeds_by_arm"][arm]
    ]


def test_every_fanout_flag_is_declared_by_the_runner():
    tree = ast.parse(TRAIN_RUNNER.read_text())
    declared: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                declared.add(argument.value)

    command_text = RUN_SPEC.read_text().split("command: >-", 1)[1]
    used = set(re.findall(r"--[a-z][a-z0-9-]*", command_text))
    launcher_flags = {"--nproc-per-node", "--standalone"}
    assert len(used - launcher_flags) >= 10
    assert used - launcher_flags <= declared


def test_runner_preserves_the_audited_bakeoff_endpoints():
    source = TRAIN_RUNNER.read_text()
    for endpoint in (
        '"val_ce"',
        '"val_tokens"',
        '"throughput_tok_s_steady"',
        '"throughput_tok_s_steady_per_device"',
        '"peak_memory_gib"',
        '"peak_memory_reserved_gib"',
        '"peak_memory_source"',
        '"flops_per_token"',
        '"mfu_pct"',
    ):
        assert endpoint in source
    assert "evaluate_val_aggregate(" in source
    assert "WARMUP_STEPS_EXCLUDED = 50" in source


def test_one_sm80_image_contains_every_required_kernel_family():
    dockerfile = DOCKERFILE.read_text()
    for pin in ("xlstm==2.0.5", "mlstm-kernels==2.0.4", "flashrnn==1.0.6"):
        assert pin in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "olmo_xlstm" in dockerfile
    assert "olmo_slstm" in dockerfile
