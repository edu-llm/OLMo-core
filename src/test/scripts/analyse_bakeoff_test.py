"""
Tests for scripts/analyse_bakeoff.py.

EVERY TEST CALLS THE SCRIPT'S OWN FUNCTION. None of them re-derives a formula, because a test
that recomputes what the code computes passes when the code changes -- which is how this repo
shipped guards that could not fire. Where a number is checked it comes from OUTSIDE this
codebase: the pre-registration's own pre-committed values, ``moe/audit/findings/power.md``, or
a closed form that is true by construction.

The special functions are additionally validated by REDUCTION -- the k=1 Dunnett critical value
must equal the Student t quantile computed by a completely different route, the zero-ncp
non-central t must equal the central t. A reduction check is evidence about the integral rather
than about itself.

Pure arithmetic over a few hundred numbers. Nothing here imports torch and nothing here runs a
model.
"""

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyse_bakeoff", str(Path(__file__).resolve().parents[3] / "scripts" / "analyse_bakeoff.py")
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("could not load scripts/analyse_bakeoff.py")
ab = importlib.util.module_from_spec(_SPEC)
# Registered BEFORE exec_module because @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for a module that is not registered.
sys.modules["analyse_bakeoff"] = ab
_SPEC.loader.exec_module(ab)

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS_JSON = REPO_ROOT / "docs" / "mixer-bakeoff" / "seeds.json"
ARMS_SOURCE = REPO_ROOT / "src" / "olmo_core" / "nn" / "transformer" / "core6_arms.py"


# ==================== Fixtures: synthetic 18-cell sets ====================
#
# The real results do not exist yet. These are how the script gets tested before they land, and
# they are built from the FROZEN seed schedule so a cell that is not on the schedule is a test
# failure rather than a passing invention.

ARM_ORDER = ["KDA_BASE", "KDA_NOACT", "KDA_GCONV", "GDN2", "KDA_R1", "KDA_R2"]

DATA_SEEDS = {1: 210007, 2: 220014, 3: 230021}

INIT_SEEDS = {
    "KDA_BASE": {1: 110007, 2: 120014, 3: 130021},
    "KDA_NOACT": {1: 113008, 2: 123015, 3: 133022},
    "KDA_GCONV": {1: 116009, 2: 126016, 3: 136023},
    "GDN2": {1: 119010, 2: 129017, 3: 139024},
    "KDA_R1": {1: 122011, 2: 132018, 3: 142025},
    "KDA_R2": {1: 125012, 2: 135019, 3: 145026},
}

L0_PARAM_TARGET = 390_135_552
ARM_DELTA = {
    "KDA_BASE": -10_080,
    "KDA_NOACT": -10_080,
    "KDA_GCONV": 2_208,
    "GDN2": 22_688,
    "KDA_R1": -10_080,
    "KDA_R2": 6_304,
}

DECLARED_STEPS = 1907
VAL_TOKENS_DECLARED = 229_894_171

# Arm-major cell index, per .edullm/run-bakeoff.yaml.
CELL_INDEX = {arm: 3 * i for i, arm in enumerate(ARM_ORDER)}

# A plausible CE surface: base ~2.5500, arms a few thousandths apart, seed noise ~0.004 nats.
BASE_CE = {
    "KDA_BASE": 2.5500,
    "KDA_NOACT": 2.5540,
    "KDA_GCONV": 2.5480,
    "GDN2": 2.5455,
    "KDA_R1": 2.5510,
    "KDA_R2": 2.5495,
}
SEED_NOISE = {1: -0.0035, 2: 0.0008, 3: 0.0027}

BASE_TPUT = {
    "KDA_BASE": 300_000.0,
    "KDA_NOACT": 305_000.0,
    "KDA_GCONV": 288_000.0,
    "GDN2": 296_000.0,
    "KDA_R1": 262_000.0,
    "KDA_R2": 148_000.0,  # the non-chunked sequential Triton kernel, ~2x slower
}
BASE_MEM = {
    "KDA_BASE": 61.0,
    "KDA_NOACT": 60.8,
    "KDA_GCONV": 61.4,
    "GDN2": 63.2,
    "KDA_R1": 62.1,
    "KDA_R2": 68.9,
}


def make_cell(
    arm: str,
    replicate: int,
    *,
    val_ce: Optional[float] = None,
    first_loss: float = 11.5203,
    steps: int = DECLARED_STEPS,
    parameters: Optional[int] = None,
    val_tokens_present: Optional[int] = VAL_TOKENS_DECLARED,
    throughput: bool = True,
    memory_source: str = "per_step_running_max",
    steady_state_steps: int = 1857,
    data_seed: Optional[int] = None,
    init_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """One synthetic per-cell summary shaped exactly like ``summarise()``'s output."""
    if val_ce is None:
        val_ce = BASE_CE[arm] + SEED_NOISE[replicate]
    if parameters is None:
        parameters = L0_PARAM_TARGET + ARM_DELTA[arm]
    cell: Dict[str, Any] = {
        "run_id": f"run_{arm.lower()}_r{replicate}",
        "dataset_id": "olmo-150b-dolma2",
        "dataset_version": "v1",
        "data_seed": DATA_SEEDS[replicate] if data_seed is None else data_seed,
        "init_seed": INIT_SEEDS[arm][replicate] if init_seed is None else init_seed,
        "gpu": "NVIDIA A100-SXM4-80GB",
        "torch": "2.7.0",
        "cuda": "12.4",
        "parameters": parameters,
        "steps": steps,
        "first_loss": first_loss,
        "last_loss": 2.61,
        "seconds": 9400.0,
        "world_size": 8,
        "arm": arm,
        "tokens_trained": steps * 524288,
        "val_ce": val_ce,
        "val_tokens": None if val_tokens_present is None else val_tokens_present - 4000,
        "val_tokens_present": val_tokens_present,
        "val_tokens_declared": VAL_TOKENS_DECLARED,
        "val_nll_sum": 5.8e8,
        "val_shards": 4,
        "sliced_eval": None,
        "checkpoint_uri": f"s3://ckpt/{arm}/r{replicate}",
        "wandb_project": "eduLLM/test",
        "wandb_url": None,
    }
    if throughput:
        total = BASE_TPUT[arm] * (1.0 + 0.002 * (replicate - 2))
        cell.update(
            {
                "throughput_tok_s_steady": total,
                "throughput_tok_s_steady_per_device": total / 8.0,
                "throughput_tok_s_whole_run": total * 0.72,
                "throughput_tok_s_whole_run_per_device": total * 0.72 / 8.0,
                "throughput_tok_s_all_steps": total * 0.96,
                "steps_measured": steps,
                "steady_state_steps": steady_state_steps,
                "warmup_steps_excluded": 50,
                "tokens_in_steady_window": steady_state_steps * 524288,
                "step_time_s_p50": 524288.0 / total,
                "step_time_s_p90": 524288.0 / total * 1.08,
                "steady_window_seconds": steady_state_steps * 524288.0 / total,
                "training_seconds_excluding_startup": steps * 524288.0 / total,
                "mfu_pct": 34.2,
                "mfu_basis": "measured",
                "device_peak_bf16_flops": 312_000_000_000_000,
                "flops_per_token": 2_400_000_000,
            }
        )
    if memory_source == "unavailable":
        cell.update(
            {
                "peak_memory_gib": None,
                "peak_memory_reserved_gib": None,
                "peak_memory_source": "unavailable",
                "peak_memory_samples": 0,
            }
        )
    else:
        cell.update(
            {
                "peak_memory_gib": BASE_MEM[arm] + 0.05 * replicate,
                "peak_memory_reserved_gib": BASE_MEM[arm] + 2.0,
                "peak_memory_source": memory_source,
                "peak_memory_samples": 0 if memory_source == "final_step_only" else steps,
            }
        )
    return cell


def happy_path_cells() -> List[Dict[str, Any]]:
    return [make_cell(arm, r) for arm in ARM_ORDER for r in (1, 2, 3)]


def write_cells(
    tmp_path: Path, cells: List[Dict[str, Any]], *, with_cell_dirs: bool = False
) -> Path:
    """Write cells to disk the way the platform lays them out."""
    root = tmp_path / "results"
    root.mkdir(exist_ok=True)
    for i, cell in enumerate(cells):
        if with_cell_dirs:
            index = CELL_INDEX[cell["arm"]] + (
                DATA_SEEDS_INVERSE.get(cell["data_seed"], 1) - 1
            )
            directory = root / f"cell-{index}"
            directory.mkdir(exist_ok=True)
            path = directory / "summary.json"
        else:
            path = root / f"{cell['run_id']}_{i}.json"
        path.write_text(json.dumps(cell, indent=2), encoding="utf-8")
    return root


DATA_SEEDS_INVERSE = {v: k for k, v in DATA_SEEDS.items()}


def run_analysis(cells: List[Dict[str, Any]], *, sources: Optional[List[str]] = None):
    """Load synthetic dicts through the script's own Cell/analyse path."""
    loaded = [
        ab.Cell(
            cell_id=c["run_id"],
            source=(sources[i] if sources else f"<memory>/{c['run_id']}.json"),
            raw=c,
            cell_index=(ab.cell_index_from_path(sources[i]) if sources else None),
        )
        for i, c in enumerate(cells)
    ]
    return ab.analyse(
        loaded,
        schedule=ab.load_seed_schedule(SEEDS_JSON),
        ledger=ab.load_arm_param_targets(ARMS_SOURCE),
    )


# ==================== Special functions ====================


class TestSpecialFunctions:
    def test_regularized_incomplete_beta_matches_closed_form(self):
        """I_x(1, 1) = x, and I_x(a, b) = 1 - I_{1-x}(b, a). Both true by construction."""
        for x in (0.05, 0.3, 0.5, 0.77, 0.99):
            assert ab.regularized_incomplete_beta(x, 1.0, 1.0) == pytest.approx(x, abs=1e-12)
        for x, a, b in ((0.3, 2.5, 4.0), (0.61, 0.5, 6.0), (0.9, 7.0, 1.5)):
            left = ab.regularized_incomplete_beta(x, a, b)
            right = 1.0 - ab.regularized_incomplete_beta(1.0 - x, b, a)
            assert left == pytest.approx(right, abs=1e-12)

    def test_regularized_gamma_p_matches_closed_form(self):
        """P(1, x) = 1 - exp(-x) exactly."""
        for x in (0.1, 1.0, 4.5, 20.0):
            assert ab.regularized_gamma_p(1.0, x) == pytest.approx(
                1.0 - math.exp(-x), abs=1e-12
            )

    def test_chi2_sf_matches_closed_form_at_df_2(self):
        """chi^2 with 2 df is Exponential(1/2): sf(x) = exp(-x/2)."""
        for x in (0.5, 3.0, 9.21, 25.0):
            assert ab.chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2.0), abs=1e-12)

    def test_chi2_ppf_inverts_chi2_sf(self):
        for p in (0.025, 0.5, 0.9, 0.975):
            for df in (1, 4, 12, 40):
                x = ab.chi2_ppf(p, df)
                assert 1.0 - ab.chi2_sf(x, df) == pytest.approx(p, abs=1e-9)

    def test_student_t_sf_matches_closed_form_at_df_1(self):
        """Cauchy: sf(t) = 1/2 - atan(t)/pi."""
        for t in (-2.0, -0.3, 0.0, 1.0, 5.0):
            assert ab.student_t_sf(t, 1) == pytest.approx(
                0.5 - math.atan(t) / math.pi, abs=1e-11
            )

    def test_student_t_ppf_reproduces_published_quantiles(self):
        """Table values quoted in the pre-registration and in power.md."""
        assert ab.student_t_ppf(0.975, 12) == pytest.approx(2.1788, abs=5e-5)
        assert ab.student_t_ppf(0.975, 4) == pytest.approx(2.7764, abs=5e-5)
        assert ab.student_t_ppf(0.975, 2) == pytest.approx(4.3027, abs=5e-5)

    def test_f_sf_matches_student_t_squared(self):
        """F(1, df) is t(df)^2, so sf(t^2; 1, df) = 2 * t_sf(|t|; df)."""
        for t, df in ((1.3, 8), (2.1788, 12), (0.4, 30)):
            assert ab.f_sf(t * t, 1, df) == pytest.approx(2.0 * ab.student_t_sf(abs(t), df),
                                                          abs=1e-11)

    def test_noncentral_t_reduces_to_central_t_at_zero_ncp(self):
        """A reduction check: ncp = 0 must give exactly the central t."""
        for t, df in ((2.9, 12), (-1.4, 7), (0.0, 3), (4.3027, 2)):
            assert ab.noncentral_t_cdf(t, df, 0.0) == pytest.approx(
                1.0 - ab.student_t_sf(t, df), abs=1e-11
            )

    def test_noncentral_t_sf_is_the_dominant_tail(self):
        assert ab.noncentral_t_sf(2.9, 12, 3.0) == pytest.approx(
            1.0 - ab.noncentral_t_cdf(2.9, 12, 3.0), abs=1e-15
        )

    def test_gauss_legendre_integrates_polynomials_exactly(self):
        """n-point Gauss-Legendre is exact to degree 2n-1. int_-1^1 x^k dx."""
        nodes, weights = ab.gauss_legendre_nodes(8)
        for k in range(0, 16):
            got = sum(w * x**k for x, w in zip(nodes, weights))
            want = 0.0 if k % 2 else 2.0 / (k + 1)
            assert got == pytest.approx(want, abs=1e-12)


class TestDunnett:
    def test_critical_value_reproduces_the_preregistered_2_902(self):
        """PREREGISTRATION s4.4: k = 5, df = 12, two-sided alpha = 0.05 -> 2.902."""
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        assert crit == pytest.approx(2.902, abs=1e-3)

    def test_k1_reduces_to_the_student_t_quantile(self):
        """With one comparison max-|t| is |t|; different route, must agree."""
        for df in (4, 12, 30):
            assert ab.dunnett_two_sided_critical_value(1, df, 0.05) == pytest.approx(
                ab.student_t_ppf(0.975, df), abs=1e-8
            )

    def test_critical_value_is_below_bonferroni_and_above_uncorrected(self):
        """Dunnett is uniformly less conservative than Bonferroni. Both bounds are external."""
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        assert ab.student_t_ppf(0.975, 12) < crit < ab.student_t_ppf(1.0 - 0.05 / (2 * 5), 12)

    def test_the_k1_reduction_holds_across_five_orders_of_magnitude_of_df(self):
        """
        REGRESSION. A fixed-width integration window is right at df = 12 and catastrophically
        wrong at large df, where the chi-scale density is a spike a fixed node count cannot
        resolve. The first version of this code returned 0.842 at df = 1e5 for a critical
        value whose true answer is 1.960 -- a plausible number, silently wrong, in a regime
        the df = 12 checks never visit.
        """
        for df in (2, 4, 12, 30, 100, 1_000, 10_000, 1_000_000):
            assert ab.dunnett_two_sided_critical_value(1, df, 0.05) == pytest.approx(
                ab.student_t_ppf(0.975, df), abs=1e-6
            ), df

    def test_infinite_df_limit_matches_the_normal_quantile(self):
        """At huge df the chi scale collapses to 1 and the answer is the normal quantile."""
        assert ab.dunnett_two_sided_critical_value(1, 5_000_000, 0.05) == pytest.approx(
            1.959964, abs=1e-5
        )

    def test_the_integration_window_captures_all_the_mass_at_every_df(self):
        """
        The guard that makes a mis-resolved integral loud rather than plausible. Checks 1 and 2
        can both pass on a drifted window -- a coarse and a fine quadrature of the same wrong
        window agree with each other -- so this one is not redundant with them.
        """
        for df in (1, 2, 12, 100, 10_000, 1_000_000):
            assert ab.chi_scale_mass(df, 96) == pytest.approx(1.0, abs=1e-6), df

    def test_the_mass_guard_can_actually_fire(self):
        """
        PROOF THE GUARD IS FIREABLE. The window that produced the 0.842 bug is reconstructed
        here and fed to the same mass integral; it must be rejected. A guard nobody has seen
        reject anything is not known to be a guard.
        """
        def old_window_mass(df: float) -> float:
            hi = 1.0 + 16.0 / math.sqrt(2.0 * df)  # the old fixed-width window
            nodes, weights = ab._scaled_nodes(96, 1e-9, hi)
            return sum(w * math.exp(ab._chi_scale_log_pdf(u, df)) for u, w in zip(nodes, weights))

        # It fires at every df where the old code returned a wrong critical value. The
        # quadrature both loses mass and overshoots as the spike outruns the node spacing, so
        # the deviation is two-sided; what matters is that it is detectable at the guard's own
        # 1e-6 tolerance.
        for df in (10_000, 100_000, 1_000_000):
            assert abs(old_window_mass(df) - 1.0) > 1e-6, df
        # And it does NOT fire at df = 12, where the old window was fine -- so the guard
        # discriminates rather than rejecting everything, which would be its own defect.
        assert abs(old_window_mass(12) - 1.0) < 1e-6

    def test_max_abs_t_cdf_is_monotone_in_k(self):
        """More comparisons cannot make the max smaller."""
        probs = [ab.equicorrelated_max_abs_t_cdf(2.5, k, 12) for k in (1, 2, 5, 9)]
        assert probs == sorted(probs, reverse=True)

    def test_validation_reports_all_three_checks_passing(self):
        v = ab.validate_dunnett_critical_value(5, 12, 0.05)
        assert v["check_1_quadrature_refinement"]["passed"]
        assert v["check_2_k1_reduces_to_student_t"]["passed"]
        assert v["check_3_integration_window_captures_the_mass"]["passed"]
        assert "quadrature" in v["method"].lower()

    def test_smm_at_rho_zero_exceeds_dunnett_at_rho_half(self):
        """Independent comparisons need a larger critical value than correlated ones."""
        assert ab.studentized_max_modulus_critical_value(
            5, 12
        ) > ab.dunnett_two_sided_critical_value(5, 12)


class TestPowerAndMde:
    def test_reproduces_power_md_paired_mde_to_five_significant_figures(self):
        """
        moe/audit/findings/power.md: MDE 0.03917 at n = 3 and 0.02018 at n = 5, s_delta = 0.0120,
        paired one-sample t (df = n-1, SE = s_delta / sqrt(n)), 80% power, exact nct.

        The pre-registration names reproducing this as the validation of the estimator.
        """
        for n, want in ((3, 0.03917), (5, 0.02018)):
            df = n - 1
            crit = ab.student_t_ppf(0.975, df)
            ncp = ab._bisect(
                lambda d: ab.noncentral_t_sf(crit, df, d), 0.0, 60.0, 0.80, iters=120
            )
            # 5 s.f. is what power.md prints, so 5 s.f. is what can be asserted.
            assert ncp * (0.0120 / math.sqrt(n)) == pytest.approx(want, abs=5e-6)

    def test_reproduces_the_preregistered_mde_table(self):
        """PREREGISTRATION s5: the four-cell MDE table at df = 12, n = 3."""
        crit_dunnett = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        crit_plain = ab.student_t_ppf(0.975, 12)
        for sigma, want_dunnett, want_plain in (
            (0.0019, 0.0059, 0.0047),
            (0.0105, 0.0327, 0.0262),
        ):
            assert ab.mde_for_power(sigma, 3, 3, crit_dunnett, 12) == pytest.approx(
                want_dunnett, abs=5e-5
            )
            assert ab.mde_for_power(sigma, 3, 3, crit_plain, 12) == pytest.approx(
                want_plain, abs=5e-5
            )

    def test_reproduces_the_run_2_mde(self):
        """PREREGISTRATION s5: run 2, df = 4, k = 1 -> 0.0058 / 0.0322."""
        crit = ab.student_t_ppf(0.975, 4)
        assert ab.mde_for_power(0.0019, 3, 3, crit, 4) == pytest.approx(0.0058, abs=5e-5)
        assert ab.mde_for_power(0.0105, 3, 3, crit, 4) == pytest.approx(0.0322, abs=5e-5)

    def test_the_normal_approximation_is_materially_too_optimistic_at_n_3(self):
        """
        PREREGISTRATION s5 / power.md line 422 state this and it is the reason the exact nct
        is mandated. The "2.2x" in both documents is specifically the CRITICAL-VALUE ratio
        ``t_{0.975,2} / z_{0.975} = 4.303 / 1.960``; the ratio of the resulting MDEs is 2.02,
        because the normal form also drops the beta-quantile's small-sample penalty. Both are
        asserted, each against the quantity it actually is.

        If the exact nct ever drifted toward the normal form this test goes red.
        """
        sigma, n = 0.0120, 3
        df = n - 1
        z_alpha, z_beta = 1.959964, 0.8416212

        crit = ab.student_t_ppf(0.975, df)
        assert crit / z_alpha == pytest.approx(2.2, abs=0.01)

        exact = ab.mde_for_power(sigma, n, n, crit, df) / math.sqrt(2.0)  # -> paired SE basis
        normal = (z_alpha + z_beta) * sigma / math.sqrt(n)
        assert exact / normal == pytest.approx(2.018, abs=0.01)
        assert exact > normal

    def test_power_for_effect_round_trips_the_mde(self):
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        mde = ab.mde_for_power(0.0105, 3, 3, crit, 12, 0.80)
        assert ab.power_for_effect(mde, 0.0105, 3, 3, crit, 12) == pytest.approx(0.80, abs=1e-6)

    def test_power_is_monotone_in_effect_size(self):
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        powers = [ab.power_for_effect(e, 0.0105, 3, 3, crit, 12) for e in (0.0, 0.01, 0.03, 0.1)]
        assert powers == sorted(powers)

    def test_contrast_standard_error_closed_form(self):
        assert ab.contrast_standard_error(0.0105, 3, 3) == pytest.approx(
            0.0105 * math.sqrt(2.0 / 3.0), abs=1e-15
        )

    def test_contrast_standard_error_refuses_n_zero(self):
        with pytest.raises(ValueError):
            ab.contrast_standard_error(0.01, 0, 3)

    def test_sigma_chi2_interval_reproduces_the_preregistered_bracket(self):
        """PREREGISTRATION s6: at df = 12 the interval is sigma_hat x [0.717, 1.651]."""
        lo, hi = ab.sigma_chi2_interval(12)
        assert lo == pytest.approx(0.717, abs=5e-4)
        assert hi == pytest.approx(1.651, abs=5e-4)


# ==================== Group statistics ====================


class TestGroupStatistics:
    def test_mean_sd_n_returns_none_sd_at_n_1_not_zero(self):
        """A single observation has NO variance estimate. 0.0 would read as a tight error bar."""
        result = ab.mean_sd_n([2.55])
        assert result["n"] == 1
        assert result["mean"] == pytest.approx(2.55)
        assert result["sd"] is None

    def test_mean_sd_n_empty_is_all_none(self):
        result = ab.mean_sd_n([])
        assert result == {"n": 0, "mean": None, "sd": None, "values": []}

    def test_pooled_variance_df_is_sum_of_n_minus_one(self):
        """PREREGISTRATION s4.3: error df = sum over arms of (n_i - 1)."""
        result = ab.pooled_variance([[1.0, 2.0, 3.0], [4.0, 5.0], [9.0]])
        assert result["df"] == 3
        assert result["contributing_groups"] == 2

    def test_pooled_variance_of_identical_groups_is_the_group_variance(self):
        """Closed form: pooling k copies of one group returns that group's variance."""
        group = [2.550, 2.554, 2.559]
        pooled = ab.pooled_variance([group, group, group])
        assert pooled["variance"] == pytest.approx(statistics.variance(group), abs=1e-15)

    def test_pooled_variance_is_none_when_no_group_has_two_cells(self):
        result = ab.pooled_variance([[1.0], [2.0]])
        assert result["variance"] is None
        assert result["sd"] is None
        assert result["df"] == 0

    def test_anova_f_matches_the_t_test_squared_for_two_groups(self):
        """External identity: one-way F on 2 groups equals the pooled two-sample t squared."""
        a, b = [2.550, 2.554, 2.559], [2.561, 2.566, 2.570]
        result = ab.one_way_anova([a, b])
        sp2 = (statistics.variance(a) * 2 + statistics.variance(b) * 2) / 4
        t = (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(sp2 * (1 / 3 + 1 / 3))
        assert result["f"] == pytest.approx(t * t, rel=1e-12)
        assert result["df_within"] == 4

    def test_anova_on_identical_groups_gives_f_zero(self):
        result = ab.one_way_anova([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert result["f"] == pytest.approx(0.0, abs=1e-20)
        assert result["p"] == pytest.approx(1.0, abs=1e-12)

    def test_anova_not_computable_with_one_group(self):
        result = ab.one_way_anova([[1.0, 2.0]])
        assert result["computable"] is False
        assert result["f"] is None

    def test_levene_is_zero_on_groups_with_identical_spread(self):
        """Shifting a group changes its mean, not its median-centred deviations."""
        a = [2.550, 2.554, 2.559]
        b = [v + 0.5 for v in a]
        result = ab.levene_median([a, b])
        assert result["statistic"] == pytest.approx(0.0, abs=1e-20)

    def test_levene_detects_a_ten_fold_spread_difference(self):
        tight = [2.5500, 2.5501, 2.5502, 2.5503]
        wide = [2.50, 2.55, 2.60, 2.70]
        result = ab.levene_median([tight, wide])
        assert result["p"] < 0.05

    def test_levene_is_median_centred_not_mean_centred(self):
        """
        PREREGISTRATION s4.5 names the MEDIAN-CENTRED (Brown-Forsythe) form, because it is the
        robust one. On a skewed group the two disagree, and centring on the mean quietly
        substitutes a different, less robust test under the same name.

        Found by mutation: swapping median for fmean left all 127 other tests green, because
        every fixture was symmetric and the two centres coincided.
        """
        skewed = [2.5500, 2.5502, 2.5504, 2.5506, 2.6900]
        plain = [2.5500, 2.5510, 2.5520, 2.5530, 2.5540]
        assert statistics.median(skewed) != pytest.approx(statistics.fmean(skewed), abs=1e-9)

        median_centred = ab.levene_median([skewed, plain])
        mean_centred = ab.one_way_anova(
            [[abs(v - statistics.fmean(g)) for v in g] for g in (skewed, plain)]
        )
        assert median_centred["statistic"] != pytest.approx(mean_centred["f"], rel=1e-3)
        # And the median-centred value is the one this function must return.
        expected = ab.one_way_anova(
            [[abs(v - statistics.median(g)) for v in g] for g in (skewed, plain)]
        )
        assert median_centred["statistic"] == pytest.approx(expected["f"], rel=1e-12)

    def test_bartlett_is_zero_on_equal_variances(self):
        a = [2.550, 2.554, 2.559]
        b = [v + 0.5 for v in a]
        result = ab.bartlett([a, b])
        assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
        assert result["p"] == pytest.approx(1.0, abs=1e-12)

    def test_bartlett_refuses_a_zero_variance_group(self):
        result = ab.bartlett([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
        assert result["computable"] is False
        assert result["statistic"] is None

    def test_welch_anova_equals_welch_t_squared_for_two_groups(self):
        """External identity, and it exercises the Satterthwaite df."""
        a, b = [2.550, 2.554, 2.559], [2.561, 2.590, 2.620]
        result = ab.welch_anova([a, b])
        va, vb = statistics.variance(a), statistics.variance(b)
        t = (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(va / 3 + vb / 3)
        df = (va / 3 + vb / 3) ** 2 / ((va / 3) ** 2 / 2 + (vb / 3) ** 2 / 2)
        assert result["f"] == pytest.approx(t * t, rel=1e-10)
        assert result["df_within"] == pytest.approx(df, rel=1e-10)


class TestContrasts:
    def test_dunnett_estimate_is_the_difference_of_means(self):
        values = {arm: [BASE_CE[arm] + SEED_NOISE[r] for r in (1, 2, 3)] for arm in ARM_ORDER}
        result = ab.dunnett_contrasts(values, "KDA_BASE")
        control_mean = statistics.fmean(values["KDA_BASE"])
        by_arm = {row["arm"]: row for row in result["contrasts"]}
        for arm in ARM_ORDER:
            if arm == "KDA_BASE":
                continue
            assert by_arm[arm]["estimate"] == pytest.approx(
                statistics.fmean(values[arm]) - control_mean, abs=1e-15
            )

    def test_dunnett_ci_half_width_is_crit_times_se(self):
        values = {arm: [BASE_CE[arm] + SEED_NOISE[r] for r in (1, 2, 3)] for arm in ARM_ORDER}
        result = ab.dunnett_contrasts(values, "KDA_BASE")
        for row in result["contrasts"]:
            assert row["ci_half_width"] == pytest.approx(
                result["critical_value"] * row["se"], abs=1e-14
            )
            assert row["ci_high"] - row["ci_low"] == pytest.approx(
                2 * row["ci_half_width"], abs=1e-14
            )

    def test_dunnett_uses_the_preregistered_k5_df12_critical_value(self):
        values = {arm: [BASE_CE[arm] + SEED_NOISE[r] for r in (1, 2, 3)] for arm in ARM_ORDER}
        result = ab.dunnett_contrasts(values, "KDA_BASE")
        assert result["k"] == 5
        assert result["df"] == 12
        assert result["rho"] == 0.5
        assert result["critical_value"] == pytest.approx(2.902, abs=1e-3)

    def test_dunnett_adjusted_p_crosses_alpha_exactly_at_the_critical_value(self):
        """A contrast whose |t| equals the critical value must have adjusted p = alpha."""
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        p = 1.0 - ab.equicorrelated_max_abs_t_cdf(crit, 5, 12, 0.5)
        assert p == pytest.approx(0.05, abs=1e-6)

    def test_dunnett_refuses_a_control_with_one_cell(self):
        """A contrast against a control with no variance estimate is not a contrast."""
        values = {"KDA_BASE": [2.55], "GDN2": [2.54, 2.55, 2.56]}
        result = ab.dunnett_contrasts(values, "KDA_BASE")
        assert result["computable"] is False
        assert "n=1" in result["reason"]

    def test_dunnett_refuses_a_missing_control(self):
        result = ab.dunnett_contrasts({"GDN2": [2.54, 2.55, 2.56]}, "KDA_BASE")
        assert result["computable"] is False

    def test_dunnett_flags_an_unbalanced_design_as_approximate(self):
        values = {
            "KDA_BASE": [2.550, 2.554, 2.559],
            "GDN2": [2.545, 2.548],
            "KDA_R1": [2.551, 2.556, 2.560],
        }
        result = ab.dunnett_contrasts(values, "KDA_BASE")
        assert result["rho"] != 0.5
        assert "APPROXIMATE" in result["rho_note"]

    def test_welch_t3_uses_the_studentized_maximum_modulus(self):
        values = {
            "KDA_BASE": [2.5500, 2.5501, 2.5502],
            "GDN2": [2.50, 2.55, 2.61],
            "KDA_R1": [2.548, 2.552, 2.556],
        }
        result = ab.welch_t3_contrasts(values, "KDA_BASE")
        assert result["computable"] is True
        assert "T3" in result["procedure"]
        for row in result["contrasts"]:
            if row.get("computable"):
                assert row["critical_value"] == pytest.approx(
                    ab.studentized_max_modulus_critical_value(2, row["welch_df"]), abs=1e-8
                )

    def test_exploratory_gate_contrast_is_gconv_minus_noact(self):
        """
        The gate is isolated by KDA_GCONV - KDA_NOACT, never KDA_GCONV - KDA_BASE: with
        gate_structure='depthwise' the pre-gate is a SiLU with a learnable slope.
        """
        values = {arm: [BASE_CE[arm] + SEED_NOISE[r] for r in (1, 2, 3)] for arm in ARM_ORDER}
        pooled = ab.pooled_variance(list(values.values()))
        result = ab.exploratory_pairwise(
            values, "KDA_GCONV", "KDA_NOACT", pooled["sd"], float(pooled["df"])
        )
        assert result["label"] == "KDA_GCONV - KDA_NOACT"
        assert "EXPLORATORY" in result["status"]
        assert result["estimate"] == pytest.approx(
            statistics.fmean(values["KDA_GCONV"]) - statistics.fmean(values["KDA_NOACT"]),
            abs=1e-15,
        )

    def test_ratio_to_control_is_the_mean_ratio_and_none_when_absent(self):
        values = {"KDA_BASE": [100.0, 102.0], "GDN2": [50.0, 52.0], "KDA_R2": []}
        result = ab.ratio_to_control(values, "KDA_BASE")
        assert result["GDN2"]["ratio_to_control"] == pytest.approx(
            statistics.fmean(values["GDN2"]) / statistics.fmean(values["KDA_BASE"])
        )
        assert result["KDA_BASE"]["ratio_to_control"] == pytest.approx(1.0)
        assert result["KDA_R2"]["ratio_to_control"] is None
        assert result["KDA_R2"]["ratio_unavailable_reason"]


# ==================== None handling ====================


class TestNoneIsNeverFine:
    def test_metric_state_distinguishes_absent_null_and_nan(self):
        raw = {"a": 1.5, "b": None, "c": float("nan"), "d": "x", "e": float("inf")}
        assert ab.metric_state(raw, "a") == ("ok", 1.5)
        assert ab.metric_state(raw, "b") == ("null", None)
        assert ab.metric_state(raw, "c") == ("nan", None)
        assert ab.metric_state(raw, "d") == ("nonnumeric", None)
        assert ab.metric_state(raw, "e") == ("nan", None)
        assert ab.metric_state(raw, "missing") == ("absent", None)

    def test_metric_state_never_returns_zero_for_a_missing_key(self):
        for key in ("nope", "also_nope"):
            state, value = ab.metric_state({}, key)
            assert value is None and value != 0.0
            assert state == "absent"

    def test_metric_state_rejects_bool_as_a_number(self):
        """True is an int in Python. A boolean throughput is not a throughput."""
        assert ab.metric_state({"x": True}, "x") == ("nonnumeric", None)

    def test_first_present_metric_records_which_key_supplied_the_value(self):
        raw = {"throughput_tok_s_steady": None, "tps_total_avg": 42.0}
        key, state, value = ab.first_present_metric(raw, ab.THROUGHPUT_TOTAL_STEADY_KEYS)
        assert key == "tps_total_avg"
        assert state == "ok"
        assert value == 42.0

    def test_first_present_metric_reports_null_not_absent_when_the_key_exists(self):
        key, state, value = ab.first_present_metric({"peak_memory_gib": None}, ("peak_memory_gib",))
        assert key is None and value is None
        assert state == "null"


# ==================== Admissibility ====================


class TestAdmissibility:
    def test_happy_path_admits_all_eighteen(self):
        report = run_analysis(happy_path_cells())
        assert report["coverage"]["cells_found"] == 18
        assert report["coverage"]["cells_admissible"] == 18
        assert report["admissibility"]["excluded_count"] == 0
        assert report["hard_errors"] == []
        assert report["coverage"]["partial"] is False

    def test_a_nan_val_ce_is_excluded_and_declared_with_its_cell_id(self):
        cells = happy_path_cells()
        cells[4]["val_ce"] = float("nan")
        report = run_analysis(cells)
        assert report["admissibility"]["excluded_count"] == 1
        excluded = report["admissibility"]["excluded_cells"][0]
        assert excluded["cell"] == cells[4]["run_id"]
        assert any("diverged" in r for r in excluded["reasons"])

    def test_an_out_of_band_val_ce_is_excluded(self):
        cells = happy_path_cells()
        cells[7]["val_ce"] = 11.4  # above ln(vocab) - 0.5: did not converge
        report = run_analysis(cells)
        assert report["admissibility"]["excluded_count"] == 1
        assert any(
            "plausibility band" in r
            for r in report["admissibility"]["excluded_cells"][0]["reasons"]
        )

    def test_an_out_of_band_first_loss_is_excluded(self):
        cells = happy_path_cells()
        cells[2]["first_loss"] = 7.4
        report = run_analysis(cells)
        assert any(
            "uniform distribution" in r
            for c in report["admissibility"]["excluded_cells"]
            for r in c["reasons"]
        )

    def test_first_loss_band_edges_are_inclusive_and_just_outside_is_excluded(self):
        for value, admissible in ((11.016, True), (12.016, True), (11.015, False), (12.017, False)):
            cells = happy_path_cells()
            cells[0]["first_loss"] = value
            report = run_analysis(cells)
            assert (report["admissibility"]["excluded_count"] == 0) is admissible, value

    def test_a_val_tokens_mismatch_is_excluded_with_no_tolerance(self):
        cells = happy_path_cells()
        cells[9]["val_tokens_present"] = VAL_TOKENS_DECLARED - 1
        report = run_analysis(cells)
        assert report["admissibility"]["excluded_count"] == 1
        assert any(
            "whole declared partition" in r
            for r in report["admissibility"]["excluded_cells"][0]["reasons"]
        )

    def test_a_short_run_is_excluded(self):
        cells = happy_path_cells()
        cells[11]["steps"] = 1400
        report = run_analysis(cells)
        assert any(
            "prefix of the data stream" in r
            for c in report["admissibility"]["excluded_cells"]
            for r in c["reasons"]
        )


class TestRealisedBudget:
    """
    THE STEP BUDGET IS A SUBMIT-TIME DECISION AND IT MOVED ONCE ALREADY (1,907 -> 1,144 steps,
    forced by A100 capacity). Admissibility must follow what the cells actually ran, not what
    seeds.json froze -- pinning to the plan would exclude all 18 as short runs and report a
    confident empty study.
    """

    def test_a_uniformly_cut_budget_admits_every_cell(self):
        cut_steps = 1144
        cells = [
            make_cell(arm, r, steps=cut_steps) for arm in ARM_ORDER for r in (1, 2, 3)
        ]
        report = run_analysis(cells)
        assert report["coverage"]["cells_admissible"] == 18
        assert report["admissibility"]["excluded_count"] == 0
        bud = report["realised_budget"]
        assert bud["steps_used_for_admissibility"] == cut_steps
        assert bud["steps_planned_in_seeds_json"] == DECLARED_STEPS
        assert bud["steps_differ_from_plan"] is True
        assert bud["steps_disagree_across_cells"] is False
        assert bud["realised_tokens_per_cell"] == cut_steps * 524288

    def test_the_realised_tpp_is_computed_from_the_cells_not_the_plan(self):
        cut_steps = 1144
        cells = [
            make_cell(arm, r, steps=cut_steps) for arm in ARM_ORDER for r in (1, 2, 3)
        ]
        report = run_analysis(cells)
        bud = report["realised_budget"]
        want = (cut_steps * 524288) / (L0_PARAM_TARGET + ARM_DELTA["KDA_BASE"])
        assert bud["realised_tpp"] == pytest.approx(want, rel=1e-9)
        assert bud["realised_tpp"] < 2.0  # the cut budget, not the planned 2.6
        joined = " ".join(report["recommendation"]["caveats"])
        assert f"TPP IS {want:.1f}" in joined
        assert "MORE inflated" in joined

    def test_the_budget_change_is_announced_in_the_report(self):
        cells = [make_cell(arm, r, steps=1144) for arm in ARM_ORDER for r in (1, 2, 3)]
        markdown = ab.render_markdown(run_analysis(cells))
        assert "Budget changed after pre-registration" in markdown
        assert "1,907" in markdown and "1,144" in markdown
        assert "MORE inflated, not less" in markdown

    def test_a_split_budget_is_a_loud_finding_and_the_minority_is_excluded(self):
        """A cell on a different budget consumed a different prefix and is not paired."""
        cells = [make_cell(arm, r, steps=1144) for arm in ARM_ORDER for r in (1, 2, 3)]
        cells[7]["steps"] = 1907
        cells[7]["tokens_trained"] = 1907 * 524288
        report = run_analysis(cells)
        bud = report["realised_budget"]
        assert bud["steps_disagree_across_cells"] is True
        assert bud["steps_used_for_admissibility"] == 1144  # the modal budget wins
        assert report["admissibility"]["excluded_count"] == 1
        assert report["admissibility"]["excluded_cells"][0]["cell"] == cells[7]["run_id"]
        markdown = ab.render_markdown(report)
        assert "DID NOT ALL RUN THE SAME BUDGET" in markdown

    def test_the_planned_budget_still_works_unchanged(self):
        """The 1,907-step case must not regress while supporting the cut."""
        report = run_analysis(happy_path_cells())
        bud = report["realised_budget"]
        assert bud["steps_used_for_admissibility"] == DECLARED_STEPS
        assert bud["steps_differ_from_plan"] is False
        assert "Budget changed" not in ab.render_markdown(report)

    def test_the_throughput_and_memory_caveat_is_stated_as_the_strong_axis(self):
        joined = " ".join(run_analysis(happy_path_cells())["recommendation"]["caveats"])
        assert "BUDGET-INDEPENDENT AND FULLY VALID" in joined
        assert "DIRECTIONAL WITH INFLATED MAGNITUDES" in joined
        assert "speed and memory conclusions are STRONG" in joined

    def test_a_parameter_mismatch_is_a_hard_error(self):
        cells = happy_path_cells()
        cells[10]["parameters"] = L0_PARAM_TARGET  # wrong: GDN2 should be +22,688
        report = run_analysis(cells)
        assert len(report["hard_errors"]) == 1
        assert report["hard_errors"][0]["arm"] == "GDN2"
        assert any(
            "NOT THE ARM THAT WAS DECLARED" in r for r in report["hard_errors"][0]["reasons"]
        )

    def test_every_arms_declared_parameter_count_is_accepted(self):
        """The ledger read out of core6_arms.py must accept the real numbers for all six arms."""
        ledger = ab.load_arm_param_targets(ARMS_SOURCE)
        assert ledger["available"] is True
        assert ledger["target"] == L0_PARAM_TARGET
        for arm, delta in ARM_DELTA.items():
            assert ab.expected_parameters(arm, ledger) == L0_PARAM_TARGET + delta

    def test_a_seed_pair_off_the_frozen_schedule_is_a_hard_error(self):
        cells = happy_path_cells()
        cells[0]["init_seed"] = 12536  # the entrypoint default: the flag never reached the draw
        report = run_analysis(cells)
        assert any(
            "not on the frozen run-1 schedule" in r
            for e in report["hard_errors"]
            for r in e["reasons"]
        )

    def test_a_seed_pair_bound_to_another_arm_is_a_hard_error(self):
        cells = happy_path_cells()
        cells[0]["init_seed"] = INIT_SEEDS["GDN2"][1]
        report = run_analysis(cells)
        assert any(
            "seeds.json binds" in r for e in report["hard_errors"] for r in e["reasons"]
        )

    def test_the_cell_index_directory_disagreeing_with_the_arm_is_a_hard_error(self):
        """
        The launcher's arm-major array and the run's own record must agree. An off-by-one in
        that array is an arm that ran twice while another never ran, and every cell exits zero.
        """
        cells = [make_cell("KDA_BASE", 1)]
        report = run_analysis(cells, sources=["/tmp/results/cell-9/summary.json"])
        assert any(
            "arm-major" in r and "disagree" in r
            for e in report["hard_errors"]
            for r in e["reasons"]
        )

    def test_a_matching_cell_index_produces_no_hard_error(self):
        cells = [make_cell("GDN2", 1)]
        report = run_analysis(cells, sources=["/tmp/results/cell-9/summary.json"])
        assert report["hard_errors"] == []

    def test_a_gross_outlier_is_warned_about_but_not_excluded(self):
        """
        No outlier rule was pre-registered, so the cell STAYS IN -- loudly. Inventing an
        exclusion criterion after seeing the data is what a pre-registration exists to prevent.

        The wording is asserted as well as the behaviour: a warning that reads "EXCLUDED" tells
        a reader the opposite of what the code did, and mutation showed the behavioural
        assertions alone did not catch that.
        """
        cells = happy_path_cells()
        cells[5]["val_ce"] = 3.9
        report = run_analysis(cells)
        assert report["admissibility"]["excluded_count"] == 0
        warnings = report["admissibility"]["outlier_warnings"]
        assert len(warnings) == 1
        assert "NOT excluded" in warnings[0]
        assert "no outlier rule was pre-registered" in warnings[0]
        assert "EXCLUDED" not in warnings[0].replace("NOT excluded", "")
        # The outlier is still in the arm's values -- flagged is not dropped.
        assert 3.9 in report["primary_endpoint"]["per_arm"][cells[5]["arm"]]["values"]
        markdown = ab.render_markdown(report)
        assert "NOT excluded" in markdown
        assert "**0 cell(s) excluded.**" in markdown

    def test_a_duplicate_cell_is_dropped_and_named(self):
        cells = happy_path_cells()
        cells.append(dict(cells[0]))
        report = run_analysis(cells)
        assert report["coverage"]["cells_found"] == 18
        assert any("DUPLICATE" in n for n in report["sources"]["dedup_notes"])

    def test_an_arm_with_one_admissible_cell_is_declared_as_having_no_variance(self):
        cells = happy_path_cells()
        for cell in cells:
            if cell["arm"] == "KDA_R2" and cell["init_seed"] != INIT_SEEDS["KDA_R2"][1]:
                cell["val_ce"] = float("nan")
        report = run_analysis(cells)
        assert report["admissibility"]["arms_with_no_variance_estimate"] == ["KDA_R2"]
        assert report["primary_endpoint"]["per_arm"]["KDA_R2"]["n"] == 1
        assert report["primary_endpoint"]["per_arm"]["KDA_R2"]["sd"] is None
        # And that arm contributes no df.
        assert report["primary_endpoint"]["pooled_sigma"]["df"] == 10

    def test_a_short_steady_window_warns_but_does_not_exclude_the_ce(self):
        cells = happy_path_cells()
        cells[3]["steady_state_steps"] = 20
        report = run_analysis(cells)
        assert report["coverage"]["cells_admissible"] == 18
        assert any(
            "not comparable" in w
            for entry in report["admissibility"]["warnings"]
            for w in entry["warnings"]
        )

    def test_a_final_step_only_memory_read_is_warned_about(self):
        cells = happy_path_cells()
        cells[6]["peak_memory_source"] = "final_step_only"
        report = run_analysis(cells)
        assert any(
            "LOWER BOUND" in w
            for entry in report["admissibility"]["warnings"]
            for w in entry["warnings"]
        )
        assert report["co_primary_endpoints"]["peak_memory_gib"]["sources_mixed"] is True
        assert report["co_primary_endpoints"]["peak_memory_gib"]["mixed_source_warning"]

    def test_saturation_exclusion_is_reported_as_zero_not_skipped(self):
        report = run_analysis(happy_path_cells())
        assert report["admissibility"]["saturation_excluded_count"] == 0
        assert "fail-open" in report["admissibility"]["saturation_note"]


# ==================== Endpoints ====================


class TestEndpoints:
    def test_per_cell_values_are_never_hidden(self):
        report = run_analysis(happy_path_cells())
        for arm in ARM_ORDER:
            assert len(report["primary_endpoint"]["per_arm"][arm]["values"]) == 3
            assert len(report["primary_endpoint"]["per_cell"][arm]) == 3

    def test_pooled_sigma_has_df_twelve_on_a_full_set(self):
        """PREREGISTRATION s6: pooled within-arm sigma at df = n_arms x (3-1) = 12."""
        report = run_analysis(happy_path_cells())
        assert report["primary_endpoint"]["pooled_sigma"]["df"] == 12
        assert report["primary_endpoint"]["pooled_sigma"]["chi2_interval"][
            "multiplier_low"
        ] == pytest.approx(0.717, abs=5e-4)
        assert "factor-2.3 bracket at df = 12" in ab.render_markdown(report)

    def test_the_sigma_bracket_widens_and_says_so_on_a_partial_run(self):
        """
        The bracket width must follow the REALISED df. Quoting the pre-registration's df = 12
        "factor 2.3" on a df = 5 run understates the uncertainty on the single number run 2
        gets sized from.
        """
        cells = [make_cell(arm, r) for arm in ARM_ORDER[:3] for r in (1, 2)]
        report = run_analysis(cells)
        interval = report["primary_endpoint"]["pooled_sigma"]["chi2_interval"]
        assert interval["df"] == 3
        factor = interval["multiplier_high"] / interval["multiplier_low"]
        assert factor > 2.3  # strictly wider than the full-run bracket
        markdown = ab.render_markdown(report)
        assert f"factor-{factor:.1f} bracket at df = 3" in markdown
        assert "so the bracket is wider" in markdown

    def test_throughput_ratio_to_control_is_reported(self):
        report = run_analysis(happy_path_cells())
        head = report["co_primary_endpoints"]["throughput_headline"]
        assert head["is_steady_state"] is True
        assert "throughput_tok_s_steady" in head["keys_used"]
        assert head["mixed_keys"] is False
        assert head["per_arm"]["KDA_R2"]["ratio_to_control"] == pytest.approx(
            BASE_TPUT["KDA_R2"] / BASE_TPUT["KDA_BASE"], rel=1e-6
        )
        assert head["per_arm"]["KDA_BASE"]["ratio_to_control"] == pytest.approx(1.0)

    def test_memory_ratio_to_control_is_reported(self):
        report = run_analysis(happy_path_cells())
        mem = report["co_primary_endpoints"]["peak_memory_gib"]["per_arm"]
        assert mem["KDA_R2"]["ratio_to_control"] > 1.0
        assert mem["KDA_BASE"]["ratio_to_control"] == pytest.approx(1.0)

    def test_missing_throughput_keys_fall_back_to_whole_run_with_a_flag(self):
        cells = happy_path_cells()
        for cell in cells:
            for key in (
                "throughput_tok_s_steady",
                "throughput_tok_s_steady_per_device",
                "tps_total_avg",
                "tps_device_avg",
            ):
                cell.pop(key, None)
        report = run_analysis(cells)
        head = report["co_primary_endpoints"]["throughput_headline"]
        assert head["is_steady_state"] is False
        assert "WHOLE-RUN" in head["source"]
        markdown = ab.render_markdown(report)
        assert "3.1x LOW" in markdown

    def test_no_throughput_at_all_is_absent_not_equal(self):
        cells = happy_path_cells()
        for cell in cells:
            for key in list(cell):
                if key.startswith("throughput_tok_s") or key.startswith("tps_"):
                    cell.pop(key)
        report = run_analysis(cells)
        head = report["co_primary_endpoints"]["throughput_headline"]
        assert head["per_arm"] == {}
        assert head["source"] == "unavailable"
        assert "No throughput data at all" in ab.render_markdown(report)

    def test_a_null_peak_memory_is_missing_not_zero(self):
        cells = happy_path_cells()
        for cell in cells:
            cell["peak_memory_gib"] = None
            cell["peak_memory_source"] = "unavailable"
        report = run_analysis(cells)
        assert report["co_primary_endpoints"]["peak_memory_gib"]["per_arm"] == {}
        assert len(report["co_primary_endpoints"]["peak_memory_gib"]["missing"]) == 18

    def test_a_zero_peak_memory_is_missing_not_the_best_arm(self):
        cells = happy_path_cells()
        cells[0]["peak_memory_gib"] = 0.0
        report = run_analysis(cells)
        missing = report["co_primary_endpoints"]["peak_memory_gib"]["missing"]
        assert len(missing) == 1
        assert missing[0]["state"] == "zero_is_not_a_measurement"
        assert report["co_primary_endpoints"]["peak_memory_gib"]["per_arm"]["KDA_BASE"]["n"] == 2

    def test_a_final_step_only_memory_figure_is_kept_out_of_the_headline_table(self):
        """
        `peak_memory_gib` changed meaning while keeping its name. A `final_step_only` read is
        the LAST STEP'S peak, a lower bound, and must not size hardware -- so it is gated out
        of the headline table and tabled separately rather than pooled with the real peaks.
        """
        cells = happy_path_cells()
        for cell in cells:
            if cell["arm"] == "GDN2":
                cell["peak_memory_source"] = "final_step_only"
        report = run_analysis(cells)
        mem = report["co_primary_endpoints"]["peak_memory_gib"]
        assert "GDN2" not in mem["per_arm"] or mem["per_arm"]["GDN2"]["n"] == 0
        assert mem["lower_bound_only_per_arm"]["GDN2"]["n"] == 3
        assert mem["gated_on_source"] == "per_step_running_max"
        markdown = ab.render_markdown(report)
        assert "MUST NOT size hardware" in markdown

    def test_an_unrecognised_memory_source_is_absent_not_assumed_good(self):
        cells = happy_path_cells()
        cells[0]["peak_memory_source"] = "some_new_source_nobody_reviewed"
        report = run_analysis(cells)
        mem = report["co_primary_endpoints"]["peak_memory_gib"]
        assert mem["per_arm"]["KDA_BASE"]["n"] == 2
        assert any("unrecognised" in m["state"] for m in mem["missing"])

    def test_the_two_throughput_measurements_are_cross_checked(self):
        """
        tps_device_avg starts after step 1, ours after 50, so upstream should sit slightly
        BELOW ours. A large gap means compilation leaked into upstream's average.
        """
        cells = happy_path_cells()
        for cell in cells:
            cell["tps_device_avg"] = cell["throughput_tok_s_steady_per_device"] * 0.97
        report = run_analysis(cells)
        cross = report["co_primary_endpoints"]["throughput_measurement_cross_check"]
        assert len(cross["per_cell"]) == 18
        assert cross["suspicious"] == []
        assert "as expected" in ab.render_markdown(report)

    def test_a_compilation_leak_into_upstreams_average_is_surfaced(self):
        cells = happy_path_cells()
        for cell in cells:
            cell["tps_device_avg"] = cell["throughput_tok_s_steady_per_device"] * 0.97
        # One cell where upstream's window swallowed the compile.
        cells[5]["tps_device_avg"] = cells[5]["throughput_tok_s_steady_per_device"] * 0.42
        report = run_analysis(cells)
        cross = report["co_primary_endpoints"]["throughput_measurement_cross_check"]
        assert len(cross["suspicious"]) == 1
        assert cross["suspicious"][0]["ratio_upstream_over_ours"] == pytest.approx(0.42, abs=1e-9)
        markdown = ab.render_markdown(report)
        assert "DISAGREEMENT BETWEEN THE TWO MEASUREMENTS" in markdown

    def test_upstream_reading_above_ours_is_also_flagged(self):
        """Backwards, not just large: one of the two is not measuring what its name says."""
        cells = happy_path_cells()
        for cell in cells:
            cell["tps_device_avg"] = cell["throughput_tok_s_steady_per_device"] * 1.30
        report = run_analysis(cells)
        assert (
            len(report["co_primary_endpoints"]["throughput_measurement_cross_check"]["suspicious"])
            == 18
        )

    def test_step_time_percentiles_are_read_from_the_committed_keys(self):
        report = run_analysis(happy_path_cells())
        assert report["co_primary_endpoints"]["step_time_median_s"]["KDA_R2"][
            "ratio_to_control"
        ] > 1.5

    def test_unconsumed_keys_are_listed_rather_than_dropped_silently(self):
        cells = happy_path_cells()
        for cell in cells:
            cell["some_brand_new_endpoint"] = 1.0
        report = run_analysis(cells)
        assert report["unconsumed_keys"]["some_brand_new_endpoint"] == 18
        assert "some_brand_new_endpoint" in ab.render_markdown(report)

    def test_a_trainer_reported_production_decision_is_passed_through_not_obeyed(self):
        cells = happy_path_cells()
        cells[0]["production_decision"] = {"choose": "KDA_R2"}
        report = run_analysis(cells)
        assert report["trainer_reported_production_decision"]
        # The recommendation is derived from the endpoints, not from that field.
        assert report["recommendation"]["choice"] != "KDA_R2"


# ==================== Recommendation ====================


class TestRecommendation:
    def test_unresolved_ce_falls_back_to_throughput_and_says_so(self):
        """The realistic case: differences of a few thousandths at sigma ~0.003, n = 3."""
        report = run_analysis(happy_path_cells())
        rec = report["recommendation"]
        assert rec["ce_resolved"] is False
        assert "throughput and memory" in rec["basis"]
        assert any("NOT RESOLVED AT n = 3" in line for line in rec["rationale"])
        assert rec["choice"] == "KDA_NOACT"  # the fastest arm in the fixture

    def test_a_large_resolved_ce_win_beats_throughput(self):
        cells = []
        for arm in ARM_ORDER:
            for r in (1, 2, 3):
                ce = BASE_CE["KDA_BASE"] + SEED_NOISE[r] * 0.1
                if arm == "GDN2":
                    ce -= 0.20  # far beyond any plausible MDE
                cells.append(make_cell(arm, r, val_ce=ce))
        report = run_analysis(cells)
        rec = report["recommendation"]
        assert rec["ce_resolved"] is True
        assert rec["choice"] == "GDN2"
        assert rec["basis"].startswith("CE (resolved)")

    def test_a_significant_effect_below_the_mde_does_not_count_as_resolved(self):
        """
        THE GAP THE MDE EXISTS FOR. A Dunnett CI excludes zero once |estimate| > crit*se, but
        the MDE is ncp_80*se with ncp_80 > crit -- so there is a band where a contrast is
        "significant" and still below what this design can detect at 80% power. Landing an
        effect in that band must NOT be reported as CE-resolved.

        Found by mutation: dropping the `clears` term from the recommendation left all 127
        other tests green, because every fixture sat far outside the band.
        """
        sigma = 0.0040
        crit = ab.dunnett_two_sided_critical_value(5, 12, 0.05)
        se = ab.contrast_standard_error(sigma, 3, 3)
        mde = ab.mde_for_power(sigma, 3, 3, crit, 12, 0.80)
        # Strictly inside the band: significant, but under the MDE.
        effect = 0.5 * (crit * se + mde)
        assert crit * se < effect < mde

        # Offsets (-sigma, 0, +sigma) give a sample sd of exactly sigma and a mean of exactly
        # the arm's centre, so the realised pooled sigma equals the one the band was solved on.
        cells = []
        for arm in ARM_ORDER:
            for r in (1, 2, 3):
                ce = 2.5500 + sigma * (r - 2)
                if arm == "GDN2":
                    ce -= effect
                cells.append(make_cell(arm, r, val_ce=ce))
        report = run_analysis(cells)
        assert report["primary_endpoint"]["pooled_sigma"]["sigma"] == pytest.approx(
            sigma, rel=1e-9
        )
        rows = {row["arm"]: row for row in report["primary_endpoint"]["dunnett"]["contrasts"]}
        assert rows["GDN2"]["excludes_zero"] is True  # significant...
        rec = report["recommendation"]
        assert rec["ce_resolved"] is False  # ...but NOT resolved
        assert "GDN2" in rec["unresolved_arms"]
        assert rec["resolved_better"] == []
        assert "throughput and memory" in rec["basis"]

    def test_an_arm_resolved_as_worse_is_struck_even_if_it_is_fastest(self):
        cells = []
        for arm in ARM_ORDER:
            for r in (1, 2, 3):
                ce = BASE_CE["KDA_BASE"] + SEED_NOISE[r] * 0.1
                if arm == "KDA_NOACT":
                    ce += 0.20
                cells.append(make_cell(arm, r, val_ce=ce))
        report = run_analysis(cells)
        rec = report["recommendation"]
        assert "KDA_NOACT" in {row["arm"] for row in rec["resolved_worse"]}
        assert rec["choice"] != "KDA_NOACT"  # despite being the fastest in the fixture
        assert any("Struck from consideration" in line for line in rec["rationale"])

    def test_no_ce_and_no_throughput_defaults_to_the_control(self):
        cells = happy_path_cells()
        for cell in cells:
            for key in list(cell):
                if key.startswith("throughput_tok_s") or key.startswith("tps_"):
                    cell.pop(key)
        report = run_analysis(cells)
        assert report["recommendation"]["choice"] == "KDA_BASE"
        assert "default to the control" in report["recommendation"]["basis"]

    def test_the_caveats_carry_tpp_the_overstatement_and_the_bound(self):
        rec = run_analysis(happy_path_cells())["recommendation"]
        joined = " ".join(rec["caveats"])
        assert "2.6" in joined
        assert "0.0103" in joined and "0.0059" in joined
        assert "BOUND, NOT EQUIVALENCE" in joined

    def test_the_mde_is_reported_with_the_recommendation(self):
        rec = run_analysis(happy_path_cells())["recommendation"]
        assert rec["mde_nats"] is not None and rec["mde_nats"] > 0.0

    def test_deviations_from_the_preregistration_are_declared(self):
        report = run_analysis(happy_path_cells())
        joined = " ".join(report["deviations_from_preregistration"])
        assert "Games-Howell" in joined
        assert "T3" in joined


# ==================== Partial input ====================


class TestPartialInput:
    def test_twelve_of_eighteen_analyses_and_names_what_is_missing(self):
        """The realistic case at 10:00: a wave still running."""
        cells = [make_cell(arm, r) for arm in ARM_ORDER[:4] for r in (1, 2, 3)]
        report = run_analysis(cells)
        assert report["coverage"]["cells_found"] == 12
        assert report["coverage"]["partial"] is True
        assert len(report["coverage"]["missing_cells"]) == 6
        missing_arms = {c["arm"] for c in report["coverage"]["missing_cells"]}
        assert missing_arms == {"KDA_R1", "KDA_R2"}
        assert report["primary_endpoint"]["pooled_sigma"]["df"] == 8
        assert report["primary_endpoint"]["dunnett"]["k"] == 3
        markdown = ab.render_markdown(report)
        assert "PARTIAL INPUT" in markdown
        assert "KDA_R1" in markdown

    def test_nine_of_eighteen_with_ragged_arms_reports_n_per_arm(self):
        """
        THE EXPECTED 10:00 STATE. Arm-major fan-out + a 61-minute median queue wait means some
        arms are complete, one is half done, and others have not started. Every arm's n must
        be stated, and the n=1 arm must not be printed as a mean with an sd.
        """
        cells = (
            [make_cell("KDA_BASE", r) for r in (1, 2, 3)]
            + [make_cell("KDA_NOACT", r) for r in (1, 2, 3)]
            + [make_cell("KDA_GCONV", r) for r in (1, 2)]
            + [make_cell("GDN2", 1)]
        )
        report = run_analysis(cells)
        census = {row["arm"]: row for row in report["coverage"]["arm_census"]}
        assert census["KDA_BASE"]["n_admissible"] == 3
        assert census["KDA_GCONV"]["n_admissible"] == 2
        assert census["GDN2"]["n_admissible"] == 1
        assert census["KDA_R1"]["n_admissible"] == 0
        assert census["KDA_R2"]["n_admissible"] == 0
        # Only arms with n >= 2 contribute df: 2 + 2 + 1 + 0 = 5.
        assert census["GDN2"]["contributes_variance"] is False
        assert report["primary_endpoint"]["pooled_sigma"]["df"] == 5
        assert report["primary_endpoint"]["per_arm"]["GDN2"]["sd"] is None

        markdown = ab.render_markdown(report)
        assert "Where each arm stands" in markdown
        assert "single obs, NOT a mean" in markdown
        assert "has exactly ONE admissible cell" in markdown
        assert "has NO admissible cells" in markdown
        assert "Absent is not the same as tied." in markdown

    def test_an_n1_arm_gets_no_contrast_row(self):
        """An arm with one cell has no CI, so it must not appear as a Dunnett contrast."""
        cells = [make_cell("KDA_BASE", r) for r in (1, 2, 3)] + [make_cell("GDN2", 1)]
        report = run_analysis(cells)
        dunnett = report["primary_endpoint"]["dunnett"]
        rows = {row["arm"]: row for row in dunnett["contrasts"]}
        assert set(rows) == {"GDN2"}
        # It is present, but with n = 1 and an honest CI, never as an sd-bearing mean.
        assert rows["GDN2"]["n"] == 1
        assert report["primary_endpoint"]["per_arm"]["GDN2"]["sd"] is None

    def test_a_single_arm_produces_a_report_rather_than_a_crash(self):
        report = run_analysis([make_cell("KDA_BASE", r) for r in (1, 2, 3)])
        assert report["primary_endpoint"]["dunnett"]["computable"] is False
        assert report["recommendation"]["choice"] is not None
        assert ab.render_markdown(report)

    def test_a_single_cell_produces_a_report_rather_than_a_crash(self):
        report = run_analysis([make_cell("KDA_BASE", 1)])
        assert report["primary_endpoint"]["pooled_sigma"]["sigma"] is None
        assert report["primary_endpoint"]["anova"]["computable"] is False
        assert report["admissibility"]["arms_with_no_variance_estimate"] == ["KDA_BASE"]
        markdown = ab.render_markdown(report)
        assert "CANNOT contribute a variance estimate" in markdown

    def test_the_control_arm_missing_entirely_is_reported_not_crashed(self):
        cells = [make_cell(arm, r) for arm in ARM_ORDER[1:] for r in (1, 2, 3)]
        report = run_analysis(cells)
        assert report["primary_endpoint"]["dunnett"]["computable"] is False
        assert "KDA_BASE" in report["admissibility"]["arms_with_no_admissible_cells"]
        assert ab.render_markdown(report)

    def test_every_cell_inadmissible_still_renders(self):
        cells = happy_path_cells()
        for cell in cells:
            cell["val_ce"] = float("nan")
        report = run_analysis(cells)
        assert report["coverage"]["cells_admissible"] == 0
        assert report["admissibility"]["excluded_count"] == 18
        assert ab.render_markdown(report)


# ==================== Loading, IO, CLI ====================


class TestLoading:
    def test_loads_a_directory_of_json_files(self, tmp_path):
        root = write_cells(tmp_path, happy_path_cells())
        cells, notes = ab.load_cells([str(root)])
        assert len(cells) == 18
        assert notes == []

    def test_loads_summaries_embedded_in_a_log_stream(self, tmp_path):
        """The platform reads results back out of the LOG STREAM, not out of a tidy file."""
        log = tmp_path / "cell.log"
        body = json.dumps(make_cell("KDA_BASE", 1), indent=2)
        log.write_text(
            "2026-08-08 09:00:00 - INFO - trainer - starting\n"
            "{\"not\": \"a cell\"}\n" + body + "\n2026-08-08 09:59:00 - INFO - done\n",
            encoding="utf-8",
        )
        cells, _ = ab.load_cells([str(log)])
        assert len(cells) == 1
        assert cells[0].raw["arm"] == "KDA_BASE"

    def test_cell_index_is_recovered_from_the_directory_name(self, tmp_path):
        root = write_cells(tmp_path, happy_path_cells(), with_cell_dirs=True)
        cells, _ = ab.load_cells([str(root)])
        assert len(cells) == 18
        assert sorted(c.cell_index for c in cells) == list(range(18))

    def test_cell_index_from_path_handles_absence(self):
        assert ab.cell_index_from_path("/tmp/results/summary.json") is None
        assert ab.cell_index_from_path("/tmp/cell-17/summary.json") == 17
        assert ab.cell_index_from_path("/tmp/cell-abc/summary.json") is None

    def test_a_missing_input_path_raises_rather_than_analysing_nothing(self):
        with pytest.raises(FileNotFoundError):
            ab.load_cells(["/definitely/not/here"])

    def test_seed_schedule_loads_and_is_frozen(self):
        schedule = ab.load_seed_schedule(SEEDS_JSON)
        assert schedule["available"] is True
        assert len(schedule["cells"]) == 18
        assert schedule["arms"] == ARM_ORDER
        assert schedule["locked"]["steps"] == DECLARED_STEPS

    def test_a_missing_seed_schedule_marks_checks_unchecked_not_passed(self, tmp_path):
        schedule = ab.load_seed_schedule(tmp_path / "nope.json")
        assert schedule["available"] is False
        assert "UNCHECKED" in schedule["reason"]
        report = ab.analyse(
            [ab.Cell(c["run_id"], "x", c) for c in happy_path_cells()],
            schedule=schedule,
            ledger=ab.load_arm_param_targets(ARMS_SOURCE),
        )
        assert any(
            "UNCHECKED" in w
            for entry in report["admissibility"]["warnings"]
            for w in entry["warnings"]
        )

    def test_an_unfrozen_seed_schedule_is_refused(self, tmp_path):
        path = tmp_path / "seeds.json"
        path.write_text(json.dumps({"status": "draft", "schedule": []}), encoding="utf-8")
        schedule = ab.load_seed_schedule(path)
        assert schedule["available"] is False
        assert "frozen" in schedule["reason"]

    def test_a_missing_arms_source_marks_the_param_audit_unchecked(self, tmp_path):
        ledger = ab.load_arm_param_targets(tmp_path / "nope.py")
        assert ledger["available"] is False
        assert ab.expected_parameters("KDA_BASE", ledger) is None

    def test_the_param_ledger_is_parsed_not_imported(self, tmp_path):
        """It must read the literals with ast, because importing would pull in torch."""
        source = tmp_path / "arms.py"
        source.write_text(
            "L0_PARAM_TARGET = 1000\nK2_L0_DELTA = -7\n"
            "ARM_L0_DELTA = {'A': 5, 'B': K2_L0_DELTA}\n",
            encoding="utf-8",
        )
        ledger = ab.load_arm_param_targets(source)
        assert ledger["target"] == 1000
        assert ab.expected_parameters("A", ledger) == 1005
        assert ab.expected_parameters("B", ledger) == 993

    def test_the_fetch_command_is_printed_and_touches_nothing(self):
        command = ab.fetch_command("s3://edullm-runs/run_019/")
        assert "aws s3 cp" in command
        assert "does not touch AWS" in command

    def test_cli_writes_json_and_prints_markdown(self, tmp_path, capsys, monkeypatch):
        root = write_cells(tmp_path, happy_path_cells())
        out = tmp_path / "bakeoff_results.json"
        monkeypatch.setattr(ab.sys.stdin, "isatty", lambda: True)
        code = ab.main([str(root), "--out", str(out)])
        assert code == 0
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["coverage"]["cells_admissible"] == 18
        assert "# Mixer bake-off" in capsys.readouterr().out

    def test_cli_exits_nonzero_on_a_hard_error(self, tmp_path, capsys, monkeypatch):
        cells = happy_path_cells()
        cells[0]["parameters"] = 1
        root = write_cells(tmp_path, cells)
        monkeypatch.setattr(ab.sys.stdin, "isatty", lambda: True)
        code = ab.main([str(root), "--out", str(tmp_path / "r.json")])
        assert code == 1
        assert "HARD ERRORS" in capsys.readouterr().out

    def test_cli_exits_two_with_no_input(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ab.sys.stdin, "isatty", lambda: True)
        assert ab.main(["--out", str(tmp_path / "r.json")]) == 2


class TestMarkdown:
    def test_the_report_names_every_excluded_cell_with_a_count(self):
        cells = happy_path_cells()
        cells[4]["val_ce"] = float("nan")
        cells[8]["steps"] = 900
        markdown = ab.render_markdown(run_analysis(cells))
        assert "**2 cell(s) excluded.**" in markdown
        assert cells[4]["run_id"] in markdown
        assert cells[8]["run_id"] in markdown

    def test_the_report_shows_the_dunnett_critical_value_and_both_sanity_checks(self):
        markdown = ab.render_markdown(run_analysis(happy_path_cells()))
        assert "2.9013" in markdown
        assert markdown.count("PASS") >= 2
        assert "quadrature" in markdown.lower()

    def test_the_report_never_presents_a_bare_p_value_for_a_contrast(self):
        """Every contrast row must carry an estimate and a CI beside any p."""
        markdown = ab.render_markdown(run_analysis(happy_path_cells()))
        assert "estimate (nats) | Dunnett 95% CI" in markdown
        assert "the honest statement is the BOUND" in markdown

    def test_the_report_states_the_gate_contrast_is_gconv_minus_noact(self):
        markdown = ab.render_markdown(run_analysis(happy_path_cells()))
        assert "`KDA_GCONV - KDA_NOACT`" in markdown
        assert "EXPLORATORY" in markdown

    def test_the_wired_gate_contrast_actually_subtracts_noact_not_base(self):
        """
        The label being right is not the same as the arithmetic being right. This asserts the
        NUMBER, which is only equal to GCONV - NOACT, and it is built on a fixture where
        NOACT and BASE differ so the two candidate answers cannot coincide.

        Found by mutation: repointing the wired call at KDA_BASE left all 127 other tests
        green, because only the hardcoded label was ever checked.
        """
        report = run_analysis(happy_path_cells())
        gate = report["primary_endpoint"]["exploratory_gate_contrast"]
        per_arm = report["primary_endpoint"]["per_arm"]
        want = per_arm["KDA_GCONV"]["mean"] - per_arm["KDA_NOACT"]["mean"]
        decoy = per_arm["KDA_GCONV"]["mean"] - per_arm["KDA_BASE"]["mean"]
        assert want != pytest.approx(decoy, abs=1e-9)
        assert gate["estimate"] == pytest.approx(want, abs=1e-12)
        assert gate["label"] == "KDA_GCONV - KDA_NOACT"

    def test_the_exploratory_ci_is_the_student_t_quantile_times_the_se(self):
        """
        The exploratory contrast is uncorrected, so its half-width is t_{0.975,df} * se -- NOT
        se, and not the Dunnett critical value either. Asserted against a quantile computed
        independently of the contrast code.

        Found by mutation: dropping the critical value from this specific half-width left the
        whole suite green, because only the Dunnett half-width was ever checked.
        """
        report = run_analysis(happy_path_cells())
        gate = report["primary_endpoint"]["exploratory_gate_contrast"]
        df = report["primary_endpoint"]["pooled_sigma"]["df"]
        half = 0.5 * (gate["ci_high"] - gate["ci_low"])
        assert half == pytest.approx(ab.student_t_ppf(0.975, df) * gate["se"], rel=1e-12)
        # And it is strictly wider than the bare SE, so the mutation is distinguishable.
        assert half > gate["se"] * 1.5
        # Uncorrected means it is NARROWER than the Dunnett-corrected half-width.
        assert half < report["primary_endpoint"]["dunnett"]["critical_value"] * gate["se"]

    def test_the_report_is_valid_markdown_tables_with_matching_columns(self):
        markdown = ab.render_markdown(run_analysis(happy_path_cells()))
        lines = markdown.splitlines()
        for i, line in enumerate(lines):
            if set(line.replace(" ", "")) <= set("|-:") and line.count("|") >= 2:
                header = lines[i - 1]
                assert header.count("|") == line.count("|"), f"line {i}: {header!r}"


class TestHomogeneityFallback:
    def test_levene_rejecting_engages_the_pre_registered_fallback(self):
        """One arm with 40x the spread of the others. Levene must reject and T3 must engage."""
        cells = []
        for arm in ARM_ORDER:
            for r in (1, 2, 3):
                spread = 0.20 if arm == "KDA_R2" else 0.0005
                ce = BASE_CE[arm] + spread * (r - 2)
                cells.append(make_cell(arm, r, val_ce=ce))
        report = run_analysis(cells)
        hom = report["primary_endpoint"]["homogeneity"]
        assert hom["rejected_at_alpha"] is True
        assert hom["fallback_engaged"] is True
        fallback = report["primary_endpoint"]["welch_fallback"]
        assert fallback["welch_anova"]["computable"] is True
        assert fallback["t3_contrasts"]["computable"] is True
        markdown = ab.render_markdown(report)
        assert "Levene REJECTS" in markdown
        assert "Welch" in markdown

    def test_levene_not_rejecting_reports_the_low_power_caveat(self):
        markdown = ab.render_markdown(run_analysis(happy_path_cells()))
        assert "almost no power" in markdown
