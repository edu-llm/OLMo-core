import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkArm:
    """One local subprocess benchmark arm with an exact backend contract."""

    name: str
    command: Tuple[str, ...]
    expected_backend: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Validated result emitted by one isolated benchmark subprocess."""

    arm: str
    pass_index: int
    position: int
    backend: str
    source_hash: str
    warmup_steps: int
    measured_steps: int
    median_step_seconds: float


def source_hash(paths: Iterable[Path]) -> str:
    """Hash path identities and bytes in deterministic order."""
    digest = hashlib.sha256()
    for path in sorted((Path(path).resolve() for path in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def nsys_profile_command(
    command: Sequence[str],
    *,
    output: Path,
    delay_seconds: int,
) -> Tuple[str, ...]:
    """Build a local post-warmup Nsight Systems command without executing it."""
    if delay_seconds < 0:
        raise ValueError("'delay_seconds' must be non-negative")
    if not command:
        raise ValueError("a profile command is required")
    return (
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        f"--delay={delay_seconds}",
        f"--output={output}",
        "--force-overwrite=true",
        *command,
    )


def counterbalanced_subprocess_benchmark(
    arms: Sequence[BenchmarkArm],
    *,
    source_paths: Sequence[Path],
    warmup_steps: int = 20,
    measured_steps: int = 50,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    extra_env: Optional[Mapping[str, str]] = None,
) -> list[BenchmarkResult]:
    """Run each arm in fresh forward/reverse subprocesses and verify its proof."""
    if not arms:
        raise ValueError("at least one benchmark arm is required")
    if warmup_steps < 20:
        raise ValueError("honest benchmark claims require at least 20 warmup steps")
    if measured_steps < 50:
        raise ValueError("honest benchmark claims require at least 50 measured steps")

    expected_source_hash = source_hash(source_paths)
    results = []
    for pass_index, ordered_arms in enumerate((tuple(arms), tuple(reversed(arms)))):
        for position, arm in enumerate(ordered_arms):
            env = os.environ.copy()
            if extra_env is not None:
                env.update(extra_env)
            env.update(
                {
                    "OLMO_BENCH_ARM": arm.name,
                    "OLMO_BENCH_EXPECTED_BACKEND": arm.expected_backend,
                    "OLMO_BENCH_SOURCE_HASH": expected_source_hash,
                    "OLMO_BENCH_WARMUP_STEPS": str(warmup_steps),
                    "OLMO_BENCH_MEASURED_STEPS": str(measured_steps),
                }
            )
            completed = runner(
                list(arm.command),
                check=True,
                capture_output=True,
                text=True,
                start_new_session=True,
                env=env,
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if not lines:
                raise RuntimeError(f"benchmark arm '{arm.name}' emitted no JSON proof")
            try:
                payload = json.loads(lines[-1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"benchmark arm '{arm.name}' did not end with a JSON proof"
                ) from exc
            if payload.get("backend") != arm.expected_backend:
                raise RuntimeError(
                    f"backend proof mismatch for '{arm.name}': "
                    f"expected {arm.expected_backend!r}, got {payload.get('backend')!r}"
                )
            if payload.get("source_hash") != expected_source_hash:
                raise RuntimeError(f"source hash proof mismatch for '{arm.name}'")
            if int(payload.get("warmup_steps", -1)) < warmup_steps:
                raise RuntimeError(f"warmup proof is too short for '{arm.name}'")
            if int(payload.get("measured_steps", -1)) < measured_steps:
                raise RuntimeError(f"measurement proof is too short for '{arm.name}'")
            results.append(
                BenchmarkResult(
                    arm=arm.name,
                    pass_index=pass_index,
                    position=position,
                    backend=payload["backend"],
                    source_hash=payload["source_hash"],
                    warmup_steps=int(payload["warmup_steps"]),
                    measured_steps=int(payload["measured_steps"]),
                    median_step_seconds=float(payload["median_step_seconds"]),
                )
            )
    return results
