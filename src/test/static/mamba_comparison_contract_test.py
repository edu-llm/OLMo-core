import ast
import json
import re
import shlex
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).parents[3]
ENTRYPOINT = ROOT / ".edullm/model_arch_tests.py"
TRAIN_RUNNER = ROOT / ".edullm/train_core6_arm.py"
DEFAULT_SPEC = ROOT / ".edullm/run.yaml"
RUN_SPEC = ROOT / ".edullm/run-comparison.yaml"
FUNCTIONAL_SMOKE_SPEC = ROOT / ".edullm/run-smoke.yaml"
THROUGHPUT_SMOKE_SPEC = ROOT / ".edullm/run-throughput-smoke.yaml"
SEEDS = ROOT / "docs/mamba-comparison/seeds.json"
DOCKERFILE = ROOT / ".edullm/Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
LAUNCHER_FLAGS = {"--nproc-per-node", "--standalone"}
#: Everything the FlashRNN prewarm hands the kernel that lands in its compile cache key.
#: One artifact is compiled per combination, so a value here that is not read off the run
#: warms a shape the arm may never call.
PREWARM_SHAPE_KEYWORDS = ("batch_size", "seq_len", "n_heads", "head_dim", "kernel_dtype")


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


def _declared_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
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
    return declared


def _shell_body(spec: Path) -> str:
    command = yaml.safe_load(spec.read_text())["command"]
    outer = shlex.split(command)
    assert outer[:2] == ["bash", "-lc"]
    return outer[2]


def _flag_values(body: str) -> dict[str, str]:
    tokens = shlex.split(body)
    values: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name, separator, inline = token.partition("=")
        if separator:
            values[name] = inline
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            values[name] = tokens[index + 1]
        else:
            values[name] = ""
    return values


def _spec_program(body: str) -> Path:
    programs = [token for token in shlex.split(body) if token.endswith(".py")]
    assert len(programs) == 1, programs
    return ROOT / programs[0]


def _pip_install_options(dockerfile: str, requirement: str) -> list[str]:
    head = dockerfile[: dockerfile.index(requirement)]
    start = head.rindex("python -m pip install ")
    return [token for token in head[start:].split() if token.startswith("--")]


def _dockerignore_patterns() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _excluded_from_build_context(path: str) -> bool:
    name = PurePosixPath(path).name
    for pattern in _dockerignore_patterns():
        if pattern.startswith("**/"):
            # Docker's ``**`` spans zero or more directories, so the tail has to be tried
            # against the basename as well as the whole context-relative path.
            if fnmatch(name, pattern[3:]) or fnmatch(path, pattern[3:]):
                return True
        elif fnmatch(path, pattern):
            return True
    return False


def test_entrypoint_freezes_eight_full_architecture_arms():
    values = _literal_assignments(ENTRYPOINT)
    # EVERY ARM AFTER THE FOURTH IS APPENDED, NEVER INSERTED, and the prefixes below are the
    # whole reason why. The fan-out index selects a cell by position, so `--fanout-size 12`
    # still reproduces the original four-arm wave byte for byte, `15` the five-arm one, and
    # `24` the eight-arm one. Inserting an arm anywhere earlier renumbers every cell after it
    # and silently repoints any fan-out already described in a document or a submission.
    assert values["ARMS"] == (
        "mamba-b3",
        "xlstm",
        "mamba3-siso-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    )
    assert values["ARMS"][:4] == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd")
    assert values["ARMS"][:5] == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd", "gdn")
    assert values["ATTENTION_LAYERS"] == (3, 7, 11, 15)
    assert values["FROZEN_STEPS"] == 1144
    assert values["FROZEN_GLOBAL_BATCH_SIZE"] == 524288


def test_arm_major_three_seed_wave_is_machine_readable():
    assert RUN_SPEC.is_file()
    assert SEEDS.is_file()
    schedule = json.loads(SEEDS.read_text())
    expected_arms = [
        "mamba-b3",
        "xlstm",
        "mamba3-siso-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    ]
    assert schedule["arms"] == expected_arms
    assert schedule["replicates_per_arm"] == 3
    assert schedule["fanout_size"] == 24
    assert schedule["cell_order"] == [arm for arm in expected_arms for _ in range(3)]
    assert schedule["steps"] == 1144
    assert schedule["global_batch_size"] == 524288
    assert schedule["tokens_per_cell"] == 599_785_472
    assert schedule["target_tokens_per_parameter"] == 1.5373
    assert schedule["warmup_steps"] == 114
    assert schedule["save_interval"] == 572

    run_yaml = RUN_SPEC.read_text()
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in run_yaml
    assert "--steps 1144" in run_yaml
    assert "--warmup-steps 114" in run_yaml
    assert "--save-interval 572" in run_yaml
    assert "--global-batch-size 524288" in run_yaml

    # The decode probe is the serving-side endpoint, and it is measured. It runs on rank zero
    # inside `summarise`, after training and outside the steady-state window, so it cannot move
    # the throughput figure the arms are ranked on. Checked against the command rather than the
    # file, so the comment explaining the choice does not satisfy the assertion.
    assert "--no-decode-probe" not in shlex.split(_shell_body(RUN_SPEC))

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


def test_the_seed_schedule_restates_the_entrypoint_ledger_rather_than_a_second_one():
    """``seeds.json`` is a publication of the entrypoint's ledger, so it must not disagree.

    The run spec is checked against ``seeds.json`` above and the runner builds from
    ``model_arch_tests``, so nothing in the chain compares those two ends to each other. A
    schedule that named an init seed the entrypoint does not admit would pass every check here
    and be refused by ``parse_args`` on the machine, one cell at a time; one that named a
    parameter count the ledger no longer holds would simply be read and believed.
    """
    values = _literal_assignments(ENTRYPOINT)
    schedule = json.loads(SEEDS.read_text())

    # `ARMS` and not `ARM_ORDER`, which is an alias of it and so is not a literal this file can
    # evaluate. `test_five_seed_eight_arm_matrix_and_parser_rejects_mismatches` imports the
    # module and pins the two to each other.
    assert schedule["arms"] == list(values["ARMS"])
    assert schedule["control"] == values["ARMS"][0]
    assert schedule["fanout_size"] == len(schedule["arms"]) * schedule["replicates_per_arm"]
    assert schedule["data_seeds"] == list(values["DATA_SEEDS"][:3])
    assert schedule["reserved_data_seeds"] == list(values["DATA_SEEDS"][3:])
    assert schedule["parameter_target"] == values["PARAMETER_TARGET"]
    assert schedule["parameter_tolerance"] == values["PARAMETER_TOLERANCE"]
    assert schedule["parameters_by_arm"] == values["EXACT_PARAMETER_COUNTS"]

    issued = []
    for arm in schedule["arms"]:
        ledger = values["INIT_SEEDS_BY_ARM"][arm]
        assert schedule["init_seeds_by_arm"][arm] == [ledger[s] for s in schedule["data_seeds"]]
        assert schedule["reserved_init_seeds_by_arm"][arm] == [
            ledger[s] for s in schedule["reserved_data_seeds"]
        ]
        issued += schedule["init_seeds_by_arm"][arm]
        issued += schedule["reserved_init_seeds_by_arm"][arm]
    assert len(set(issued)) == len(issued)
    assert set(issued).isdisjoint(values["DATA_SEEDS"])


def test_every_fanout_flag_is_declared_by_the_runner():
    declared = _declared_flags(TRAIN_RUNNER)

    command_text = RUN_SPEC.read_text().split("command: >-", 1)[1]
    used = set(re.findall(r"--[a-z][a-z0-9-]*", command_text))
    assert len(used - LAUNCHER_FLAGS) >= 10
    assert used - LAUNCHER_FLAGS <= declared


def test_default_spec_is_a_short_functional_smoke_on_one_mamba_b3_cell():
    spec = yaml.safe_load(DEFAULT_SPEC.read_text())
    assert spec["schema_version"] == 1
    assert spec["workload_profile"] == "olmo-core-train"
    assert spec["suggested_compute"] == "gpu-8xa100"

    body = _shell_body(DEFAULT_SPEC)
    tokens = shlex.split(body)
    assert "$EDULLM_RUN_ID" in tokens
    assert "--nproc-per-node=8" in tokens

    values = _literal_assignments(ENTRYPOINT)
    flags = _flag_values(body)
    data_seed = int(flags["--data-seed"])
    assert flags["--arm"] == "mamba-b3"
    assert data_seed in values["DATA_SEEDS"]
    assert int(flags["--init-seed"]) == values["INIT_SEEDS_BY_ARM"]["mamba-b3"][data_seed]

    # TEN STEPS IS A BOUND, NOT A PREFERENCE. This file is what a bare submission runs, and
    # the runner's own --steps default is the frozen comparison cell, so a default command
    # that leaves the flag off bills the whole 1144-step run by omission.
    steps = int(flags["--steps"])
    assert 0 < steps <= 10
    assert steps != values["FROZEN_STEPS"]
    assert int(flags["--warmup-steps"]) < steps

    # The geometry is the measured run's, so the smoke exercises the shapes it will use.
    assert int(flags["--sequence-length"]) == 4096
    assert int(flags["--global-batch-size"]) == values["FROZEN_GLOBAL_BATCH_SIZE"]
    assert int(flags["--rank-microbatch-size"]) == 8192
    assert float(flags["--learning-rate"]) == 1.4e-3

    # At the end of the run or past it, so no mid-run checkpoint dispatch.
    assert int(flags["--save-interval"]) >= steps

    # Nothing here is measured, so neither expensive endpoint may run.
    assert flags["--skip-heldout-eval"] == ""
    assert flags["--no-decode-probe"] == ""

    # ``main()`` refuses the run when this reaches it empty.
    assert flags["--save-folder"] == "$EDULLM_CHECKPOINT_DIR"

    # The precision and Triton contracts have to be readable in the command text.
    assert flags["--param-dtype"] == "bfloat16"
    assert "TRITON_F32_DEFAULT=ieee" in body

    # An undeclared flag is not an error: parse_known_args swallows it and hands it to
    # config.merge() as a dotted override, which dies after the corpus has already loaded.
    assert set(flags) - LAUNCHER_FLAGS <= _declared_flags(TRAIN_RUNNER)
    for dispatcher in ("edullm submit", "boto3", "aws s3", "aws batch"):
        assert dispatcher not in body


def test_default_spec_runs_the_same_preflighted_runner_as_the_dedicated_smokes():
    assert _spec_program(_shell_body(DEFAULT_SPEC)) == TRAIN_RUNNER
    for spec in (FUNCTIONAL_SMOKE_SPEC, THROUGHPUT_SMOKE_SPEC, RUN_SPEC):
        assert _spec_program(_shell_body(spec)) == TRAIN_RUNNER

    tree = ast.parse(TRAIN_RUNNER.read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "preflight_accelerated_arm"
        for node in ast.walk(main)
    )

    # model_arch_tests.py builds the same four architectures and has no such guard, so a
    # default routed through it dies inside the first step with the kernel's own error,
    # after the image pull and the queue wait have already been paid for.
    assert "preflight_accelerated_arm" not in ENTRYPOINT.read_text()


def test_smoke_fanout_seeds_match_the_frozen_arm_table():
    values = _literal_assignments(ENTRYPOINT)
    data_seed = values["DATA_SEEDS"][0]
    expected_by_arm = values["INIT_SEEDS_BY_ARM"]

    for spec in (FUNCTIONAL_SMOKE_SPEC, THROUGHPUT_SMOKE_SPEC):
        text = spec.read_text()

        def shell_array(name: str) -> list[str]:
            match = re.search(rf"{name}=\((.*?)\) &&", text, flags=re.DOTALL)
            assert match is not None, (spec, name)
            return match.group(1).split()

        arms = shell_array("ARMS")
        init_seeds = [int(seed) for seed in shell_array("ISEEDS")]
        # EVERY ARM OF THE WAVE, IN THE WAVE'S OWN ORDER, rather than a count of cells.
        # A count is what let the three KDA arms be appended to the wave while both smokes
        # went on rehearsing five, which would have sent nine of the twenty-four cells to a
        # machine unrehearsed on a study with --attempts pinned to 1. Requiring the wave's
        # order too is what keeps --fanout-size 5 the same five cells it has always been.
        assert arms == list(values["ARMS"])
        assert len(init_seeds) == len(arms)
        assert init_seeds == [expected_by_arm[arm][data_seed] for arm in arms]
        assert int(_flag_values(_shell_body(spec))["--data-seed"]) == data_seed


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


def test_rank_device_is_selected_before_xlstm_prewarm():
    tree = ast.parse(TRAIN_RUNNER.read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def named_calls(statements, name):
        return [
            node
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    prepare_calls = named_calls(main.body, "prepare_training_environment")
    preflight_calls = named_calls(main.body, "preflight_accelerated_arm")
    assert len(prepare_calls) == len(preflight_calls) == 1
    assert prepare_calls[0].lineno < preflight_calls[0].lineno
    assert any(
        preflight_calls[0] in named_calls(node.body, "preflight_accelerated_arm")
        and named_calls(node.finalbody, "teardown_training_environment")
        for node in ast.walk(main)
        if isinstance(node, ast.Try)
    )


def test_the_flashrnn_prewarm_reads_its_whole_shape_and_dtype_off_the_run():
    tree = ast.parse(TRAIN_RUNNER.read_text())
    prewarms = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prewarm_flashrnn"
    ]
    assert len(prewarms) == 1

    keywords = {word.arg: word.value for word in prewarms[0].keywords}
    assert set(PREWARM_SHAPE_KEYWORDS) <= set(keywords)
    for name in PREWARM_SHAPE_KEYWORDS:
        # EVERY ONE OF THESE FIVE IS IN FLASHRNN'S CACHE KEY, so a literal here is the bug
        # itself: the arm's sLSTM layers moved to bfloat16 while this call went on asking
        # for float32, and the preflight compiled an artifact no step ever called while the
        # real kernel compiled inside the first measured one.
        #
        # "Not a Constant" is not the test, because a folded literal drifts exactly as
        # quietly as a written one: ``1024 // 4`` is an ast.BinOp and named the head
        # dimension of a four-head layer long after the arm was free to carry another.
        assert any(
            isinstance(node, (ast.Name, ast.Attribute)) for node in ast.walk(keywords[name])
        ), name


def test_smoke_specs_separate_functional_and_throughput_measurements():
    source = TRAIN_RUNNER.read_text()
    assert '"--skip-heldout-eval"' in source
    assert "if opts.skip_heldout_eval:" in source

    # One literal for both specs, because a smoke that rehearses a different set of arms from
    # the other smoke is worse than either of them alone: the functional gate would clear an
    # arm the timed one never runs, and nothing would say so.
    wave_arms = "ARMS=(mamba-b3 xlstm mamba3-siso-pd native-pd gdn kda kda-hh-r2 kda-gconv)"

    functional = FUNCTIONAL_SMOKE_SPEC.read_text()
    assert "--steps 10" in functional
    assert "--skip-heldout-eval" in functional
    assert "--no-decode-probe" in functional
    assert "--nproc-per-node=8" in functional
    assert wave_arms in functional

    throughput = THROUGHPUT_SMOKE_SPEC.read_text()
    assert "--steps 100" in throughput
    assert "--save-interval 101" in throughput
    assert "--skip-heldout-eval" in throughput
    assert "--nproc-per-node=8" in throughput
    assert "WARMUP_STEPS_EXCLUDED=50" in throughput
    assert "TORCH_LOGS=graph_breaks,recompiles" in throughput
    assert wave_arms in throughput


def test_one_sm80_image_contains_every_required_kernel_family():
    dockerfile = DOCKERFILE.read_text()
    for pin in ("xlstm==2.0.5", "mlstm-kernels==2.0.4", "flashrnn==1.0.6"):
        assert pin in dockerfile
    for package in (
        "libcublas-dev-12-8=12.8.4.1-1",
        "libcusolver-dev-12-8=11.7.3.90-1",
        "libcusparse-dev-12-8=12.5.8.93-1",
    ):
        assert package in dockerfile
    for header in ("cublas_v2.h", "cusolverDn.h", "cusparse.h"):
        assert f"/usr/local/cuda-12.8/include/{header}" in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "olmo_xlstm" in dockerfile
    assert "olmo_slstm" in dockerfile


def test_final_project_install_builds_against_the_pinned_build_tools():
    dockerfile = DOCKERFILE.read_text()

    project_install = re.search(
        r"^\s*python -m pip install ([^\n]*?) \.;", dockerfile, flags=re.MULTILINE
    )
    assert project_install is not None
    options = project_install.group(1).split()
    assert "--no-build-isolation" in options
    assert "--no-deps" in options
    assert "--no-cache-dir" in options

    for pin in ('"setuptools==69.5.1"', '"wheel==0.45.1"'):
        assert 0 <= dockerfile.index(pin) < project_install.start()

    # The install layer keeps the assertions that make a broken image fail the build.
    layer = dockerfile[project_install.start() :]
    assert "torch.version.cuda == '12.8'" in layer
    assert "'sm_80' in arch_flags" in layer
    assert "import torch, _flash_pd_native_cuda" in layer
    assert "test -s /usr/local/share/licenses/nxai/NOTICE" in layer


def test_pinned_edullm_data_install_cannot_move_the_pinned_runtime():
    dockerfile = DOCKERFILE.read_text()
    requirement = (
        '"edullm-data @ https://github.com/edu-llm/edullm-data/archive/'
        '38bf831a6c3f445e394784018441fd59288b876c.tar.gz"'
    )
    assert requirement in dockerfile

    options = _pip_install_options(dockerfile, requirement)
    assert "--no-deps" in options
    assert "--no-build-isolation" in options
    assert "--no-cache-dir" in options

    # --no-deps is only safe because everything edullm-data declares at runtime is pinned
    # above it, and --no-build-isolation only resolves because its backend is pinned there
    # too: without hatchling in the image pip fails the layer with BackendUnavailable.
    for pin in (
        '"numpy==2.3.5"',
        '"boto3==1.40.70"',
        '"hatchling==1.27.0"',
        '"pathspec==1.1.1"',
        '"pluggy==1.6.0"',
        '"trove-classifiers==2026.6.1.19"',
    ):
        assert 0 <= dockerfile.index(pin) < dockerfile.index(requirement)


def test_build_context_excludes_native_artifacts_that_would_shadow_the_wheel():
    patterns = _dockerignore_patterns()
    for suffix in ("so", "so.*", "pyd", "dylib", "o", "a"):
        assert f"**/*.{suffix}" in patterns

    # A locally built extension sits beside the source that the image puts on PYTHONPATH,
    # so COPY . . would land it ahead of the wheel built inside the image for this torch.
    assert "PYTHONPATH=/opt/olmo-core/src" in DOCKERFILE.read_text()
    for artifact in (
        "src/_flash_pd_native_cuda.cpython-312-x86_64-linux-gnu.so",
        "build/flash_pd_native_cuda_lib/libcudart.so",
        "libcudart.so.13",
    ):
        assert _excluded_from_build_context(artifact)

    # The sources the image compiles that extension from still reach the build.
    for source in (
        "flash_pd_native_setup.py",
        "src/olmo_core/nn/flash_pd_native/csrc/flash_pd_native.cpp",
        "src/olmo_core/nn/flash_pd_native/csrc/flash_pd_native_cuda.cu",
    ):
        assert not _excluded_from_build_context(source)
