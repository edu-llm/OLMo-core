"""Freeze the confirmation preregistration from actual artifacts, then verify it per replicate.

Run on FarmShare from the launch-commit snapshot, with HF_HOME set:
    python -I -B build_preregistration.py build REPORT REPLICATES_ROOT AA_REPORT REPOSITORY CODE_COMMIT OUTPUT
    python -I -B build_preregistration.py verify PREREG MANIFEST

`build` fully re-verifies the final-span report, applies the frozen 0.08 margin amendment and binds
the 24 prepared manifests, the GPU A/A report and its code-compatibility packet. `verify` runs
the same check a confirmation launch runs (without the git HEAD check, since snapshots have no .git).
"""

import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.acquisition import SOURCE_QA_ACQUISITION_POLICY  # noqa: E402
from spacing_rerun.common import digest, read_json, write_json  # noqa: E402
from spacing_rerun.confirmation import PROTOCOL, build_aa_compatibility, verify_confirmation_preregistration  # noqa: E402
from spacing_rerun.final_span import (MARGIN_AMENDMENT, MARGIN_AMENDMENT_SHA256, amended_precision_plan,  # noqa: E402
                                      verify_final_span_report)
from spacing_rerun.teaching import SOURCE_QA_POLICY  # noqa: E402
from spacing_rerun.units import GROUNDED_METRICS, GROUNDED_POLICY  # noqa: E402


def build(report_path, replicates_root, aa_path, repository, code_commit, output):
    output = pathlib.Path(output)
    if output.exists():
        sys.exit("output exists; never overwrite")
    report = verify_final_span_report(report_path)
    plan = amended_precision_plan(report)
    n = plan["chosen_n"]
    manifests = [read_json(pathlib.Path(replicates_root) / f"replicate-{i:02d}/prepared/manifest.json")
                 for i in range(1, n + 1)]
    first = manifests[0]
    audit = first["confirmation_audit"]
    summary = first["confirmation_audit_summary"]
    if any(m["confirmation_audit"]["sha256"] != audit["sha256"] or m["tokenizer_sha256"] != first["tokenizer_sha256"] or
           m["model_revision"] != first["model_revision"] for m in manifests):
        sys.exit("replicates disagree on audit, tokenizer or model revision")
    aa = read_json(aa_path)
    compatibility = build_aa_compatibility(aa, code_commit, repository)
    cost = report["cost"]
    per_bundle = cost["largest_observed_five_arm_bundle_gpu_hours"]
    prereg = {
        "schema": "spacing-confirmation-preregistration-v1",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "confirmation_protocol_schema": PROTOCOL, "rehearsal_unit_policy": GROUNDED_POLICY,
        "metric_schema": GROUNDED_METRICS, "acquisition_policy": SOURCE_QA_ACQUISITION_POLICY,
        "qa_teaching_policy": SOURCE_QA_POLICY,
        "confirmation_audit_sha256": audit["sha256"], "source_catalog_sha256": audit["source_catalog"]["sha256"],
        "audit_conformance_amendment_sha256": summary.get("audit_conformance_amendment_sha256"),
        "human_review_complete": False,
        "final_span_report_path": str(pathlib.Path(report_path).resolve()), "final_span_report_sha256": report["sha256"],
        "margin_amendment_sha256": MARGIN_AMENDMENT_SHA256, "loss_margin": MARGIN_AMENDMENT["margin"],
        "E": report["selected_E"], "n": n, "inference_claim": plan["claim"],
        "planning_policy_sha256": report["planning_policy_sha256"], "power_scenarios": plan,
        "primary_delay": 84, "common_span": 252, "buffer_steps": 168, "cohort_size": 2, "review_cap": 8,
        "secondary_holm_family": ["H1", "H3", "HG"],
        "missing_pair_rule": ("Every preregistered replicate must complete all arms within the fixed attempt budget. "
                              "The confirmation analysis refuses an incomplete replicate set (analysis.py), so a "
                              "missing pair is reported as a failed confirmation; it is never dropped or replaced."),
        "maximum_attempts": first["config"]["max_attempts"],
        "event_cluster_sensitivity_policy": "same_role_components_as_single_cluster_then_equal_clusters_v1",
        "same_role_components": summary["same_role_components"],
        "model_revision": first["model_revision"], "tokenizer_sha256": first["tokenizer_sha256"],
        "measured_cost": {
            "final_span_report_sha256": report["sha256"],
            "confirmation_projection": {
                "n": n, "arm_gpu_hours_at_largest_observed_development_bundle": round(per_bundle * n, 4),
                "basis": ("Largest observed five-arm development bundle times n. Excludes Stage-1 acquisition, "
                          "preparation, queue time and retries.")},
            "content_scale_limitations": (
                "Confirmation replicates train about 220-280 old and 350-415 new grounded groups. That is more content "
                "than the development bundles the cost was measured on, so per-arm time may be longer. Budget with a "
                "margin and keep the fixed attempt limit.")},
        "replicate_manifest_sha256": [m["sha256"] for m in manifests],
        "replicate_designs": [{"manifest_sha256": m["sha256"], "roles_sha256": digest(m["split"]["roles"]),
                               **{k: m["config"][k] for k in ("split_seed", "order_seed", "train_seed", "eval_seed")}}
                              for m in manifests],
        "aa_report_path": str(pathlib.Path(aa_path).resolve()), "aa_report_sha256": digest(aa),
        "aa_code_compatibility": compatibility, "hardware": {"gpu": aa["gpu"]}, "code_commit": code_commit,
        "main_launched": False,
    }
    prereg["sha256"] = digest(prereg)
    write_json(output, prereg)
    verify_confirmation_preregistration(prereg, first, check_code=False)
    print(json.dumps({"sha256": prereg["sha256"], "n": n, "E": prereg["E"], "claim": plan["claim"],
                      "loss_margin": prereg["loss_margin"], "gpu": aa["gpu"], "code_commit": code_commit}, indent=1))


def verify(prereg_path, manifest_path):
    prereg = read_json(prereg_path)
    verify_confirmation_preregistration(prereg, read_json(manifest_path), check_code=False)
    print(json.dumps({"verified": str(manifest_path), "prereg_sha256": prereg["sha256"]}))


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1]](*sys.argv[2:])
