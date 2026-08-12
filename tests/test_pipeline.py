"""End-to-end: generator -> masker -> bins -> trainer.

The unit tests check each stage in isolation; this checks that they join up, which
is where the previous generation's corpus went wrong. The key assertion is the
*global* equal-mass property: over a whole corpus, `split` and both controls must
mask exactly the same number of tokens.

Note what is NOT a valid check: comparing `supervised_frac` from a single training
window. The masks differ in *placement*, so local density varies between arms even
when global counts are identical. Asserting on a window is how you talk yourself
into thinking the controls are mismatched when they are exact.
"""

import numpy as np
import pytest

from memsplit import bios, masking, nhop
from memsplit.records import spans_from_roles
from memsplit.tokenizer import get_tok
from memsplit.trainer import TrainConfig, Trainer

TOK = get_tok("byte")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Build a small real corpus: one stream, four sidecars."""
    root = tmp_path_factory.mktemp("nhop_v1")
    recs = bios.generate_records(300, seed=0)
    graph = nhop.build_graph(recs, n_layers=6, seed=0)
    by_id = {r.entity_id: r for r in recs}

    stream: list[int] = []
    side: dict[str, list[int]] = {c: [] for c in masking.CONDITIONS}
    diags, docs, skipped = [], 0, 0
    for exposure in range(2):
        for eid in nhop.eligible_starts(graph, 3):
            for depth in (1, 2, 3):
                got = nhop.sample_item(graph, by_id, eid, depth, "employer", seed=0)
                if not got:
                    continue
                chain, end, value = got
                doc = nhop.render_doc(
                    graph, by_id, eid, chain, "employer", value, exposure
                )
                ids, spans = spans_from_roles(TOK, doc.segments, doc.roles)
                try:
                    plan = masking.derive_weights(spans, len(ids), seed=eid * 10 + depth)
                except masking.ControlUndersupply:
                    skipped += 1
                    continue
                stream.extend(ids)
                for cond in masking.CONDITIONS:
                    side[cond].extend(plan.weights[cond].tolist())
                diags.append(plan.diagnostics)
                docs += 1

    np.array(stream, dtype=np.uint16).tofile(root / "tokens.bin")
    for cond in masking.CONDITIONS:
        np.array(side[cond], dtype=np.uint8).tofile(root / f"weights.{cond}.bin")
    return {
        "root": root,
        "n_tokens": len(stream),
        "side": {c: np.array(v, dtype=np.uint8) for c, v in side.items()},
        "report": masking.aggregate_report(diags),
        "docs": docs,
        "skipped": skipped,
    }


def test_corpus_builds_at_scale(corpus):
    assert corpus["docs"] > 500, corpus["docs"]
    assert corpus["n_tokens"] > 100_000, corpus["n_tokens"]
    # Skipping is allowed but must be rare; a high rate means the control pool is
    # too small for the payload mass and the design needs more prose per document.
    rate = corpus["skipped"] / (corpus["docs"] + corpus["skipped"])
    assert rate < 0.25, rate


def test_global_equal_mass_is_exact(corpus):
    """split and both controls mask the same COUNT over the whole corpus."""
    z = {c: int((v == 0).sum()) for c, v in corpus["side"].items()}
    assert z["dense"] == 0
    assert z["random_contig"] == z["split"], z
    assert z["random_scatter"] == z["split"], z
    assert z["split"] > 0


def test_controls_never_overlap_a_payload_token(corpus):
    s = corpus["side"]["split"]
    for cond in ("random_contig", "random_scatter"):
        overlap = int(((s == 0) & (corpus["side"][cond] == 0)).sum())
        assert overlap == 0, (cond, overlap)


def test_every_condition_indexes_one_stream_of_equal_length(corpus):
    n = corpus["n_tokens"]
    for cond, v in corpus["side"].items():
        assert len(v) == n, (cond, len(v), n)


def test_manifest_reports_matching(corpus):
    rep = corpus["report"]
    assert rep["count_matched_contig"] and rep["count_matched_scatter"]
    assert 0.05 < rep["masked_token_frac_split"] < 0.35
    assert rep["restate_share_of_value_tokens"] > 0


def test_vocab_mismatch_fails_with_a_useful_message(corpus):
    """A 512-vocab model against a corpus with control tokens at 50257+."""
    cfg = TrainConfig(
        run_id="vocabfail", out_dir=str(corpus["root"] / "vf"),
        data_root=str(corpus["root"]), condition="dense", preset="toy",
        ctx=64, micro_batch_size=2, tokens_per_step=128, total_tokens=128,
        device="cpu", resume_required=False,
    )
    with pytest.raises(ValueError, match="model vocabulary"):
        Trainer(cfg)


def test_each_arm_trains_on_the_real_corpus(corpus):
    """Every arm completes and logs finite, well-scaled losses.

    Deliberately does NOT assert that loss decreases. At four steps, with a cosine
    schedule decaying across those same four steps, loss going up is ordinary
    noise -- asserting a decrease here would be a flaky test dressed up as a
    correctness check. Learning is covered by `test_resume_is_bit_exact`, which
    trains long enough for the trajectory to mean something.
    """
    import json
    import math

    for cond in masking.CONDITIONS:
        cfg = TrainConfig(
            run_id=f"e2e_{cond}", out_dir=str(corpus["root"] / f"run_{cond}"),
            data_root=str(corpus["root"]), condition=cond, preset="toy",
            # The window must be wide enough to span a payload span. Documents
            # average ~410 tokens under the byte tokenizer, and the first ~100 are
            # the question prompt, so a 64-token window sees no masked tokens at
            # all and every arm reports supervised_frac == 1.0.
            vocab_size=50304, ctx=128, micro_batch_size=4, tokens_per_step=512,
            total_tokens=512 * 3, lr=3e-3, warmup_steps=1, device="cpu",
            log_every=1, checkpoint_minutes=1e9, resume_required=False,
        )
        t = Trainer(cfg).train(resume=False)
        assert t.step == cfg.total_steps

        rows = [
            json.loads(l)
            for l in (t.out / "log.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert rows, cond
        for r in rows:
            assert math.isfinite(r["loss"]) and r["loss"] > 0, (cond, r)
            assert r["loss_divisor"] == cfg.loss_divisor
        # Under the fixed divisor a masked arm sums fewer terms over the same
        # denominator, so its reported loss must be BELOW the dense arm's. That is
        # correct and is exactly why the number is not a modelling comparison.
        if cond == "dense":
            assert rows[-1]["supervised_frac"] == 1.0
        else:
            assert rows[-1]["supervised_frac"] < 1.0


def test_masked_arm_reports_lower_loss_and_that_is_not_an_improvement(corpus):
    """Guards the interpretation, not the number.

    A split arm's training loss is lower than dense's by construction under a
    fixed divisor. `supervised_frac` is logged beside it so nobody reads that as
    better modelling -- the previous line's trainer normalised the difference away
    and thereby made the arms non-comparable instead.
    """
    import json

    out = {}
    for cond in ("dense", "split"):
        cfg = TrainConfig(
            run_id=f"cmp_{cond}", out_dir=str(corpus["root"] / f"cmp_{cond}"),
            data_root=str(corpus["root"]), condition=cond, preset="toy",
            vocab_size=50304, ctx=128, micro_batch_size=4, tokens_per_step=512,
            total_tokens=512 * 2, lr=1e-4, warmup_steps=1, device="cpu",
            log_every=1, checkpoint_minutes=1e9, resume_required=False,
        )
        t = Trainer(cfg).train(resume=False)
        rows = [
            json.loads(l)
            for l in (t.out / "log.jsonl").read_text().splitlines()
            if l.strip()
        ]
        out[cond] = rows[0]

    assert out["split"]["loss"] < out["dense"]["loss"]
    assert out["split"]["supervised_frac"] < out["dense"]["supervised_frac"] == 1.0
