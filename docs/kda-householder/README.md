# KDA + Householder: write-up and evidence

A per-channel-gated Householder delta-product recurrence for linear attention — the
implementation, its verification, and what the experiments do and do not establish.

| File | Contents |
|---|---|
| [`kda-householder.md`](kda-householder.md) | The write-up. Start here. |
| `data/*.tsv` | Every number in the write-up, as generated data. |
| `scripts/*.py` | The scripts that produced `data/`. Deterministic, stdlib-only where possible. |

## Provenance

The implementation under discussion is commit `6b75c06` on branch
`agent/claude-01/dp2-kda-phase-0-prep`; the probe harness is vendored at `8107642`.
Raw run artifacts live on Stanford FarmShare under
`/scratch/users/ericrcwu/kda/` and are byte-identical to the local copies used
here (md5-of-md5s `d5a89d9f115b53cc94a1329966a94da2` over the 98 probe runs).

Every statistic in the write-up was recomputed from raw per-run JSON rather than
copied from prior prose. `scripts/verify_claims.py` regenerates the probe tables and
reports claimed-vs-recomputed for each; `scripts/analyze_lm_final.py` does the same
for the language-model runs. Where a recomputed number disagrees with the earlier
project handoff, the write-up carries the recomputed value and says so explicitly in
[§9](kda-householder.md#9-corrections-to-the-project-record).

## Reproduction

All computation runs on FarmShare (L40S, sm_89). The analysis scripts are CPU-only
and need no GPU:

```bash
python scripts/verify_claims.py --results-root /scratch/users/ericrcwu/kda/probes/results
python scripts/effective_horizon.py /scratch/users/ericrcwu/kda/probes/results/all_night
python scripts/prefix_correction.py
python scripts/horizon_depth.py
python scripts/param_confound.py
python scripts/depth_ladder_check.py
python scripts/analyze_lm_final.py
```

Two things to know before re-running the *experiments*, as opposed to the analysis:

- **Anchor the probe harness at git `93b60d7`.** The copy vendored at `probes/` is a later
  evolution and rejects the original command lines (`--seed` is ambiguous against
  `--seed-init`/`--seed-data`). It fails loudly rather than silently, but it does not
  reproduce the published runs.
- **Runs are not bit-reproducible.** Re-running a published configuration at the same seed
  and backend reproduces 0/18 configurations exactly, with differences up to 10.4pp. Only
  seed-averaged quantities and paired contrasts are stable — see
  [§5.0a](kda-householder.md#50a-runs-are-not-bit-reproducible-and-this-bounds-what-counts-as-an-effect).
