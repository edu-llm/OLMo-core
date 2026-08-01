# LIV experiment — research, probes, and cluster artifacts

Supporting material for the LFM2 "LIV" (short-conv) study whose implementation lives in
`src/olmo_core/nn/attention/short_conv.py` and `src/olmo_core/nn/transformer/liv_arms.py`.

**This directory is archival, not library code.** Nothing here is imported by `olmo_core`, nothing
here runs in CI, and the scripts are not held to the package's lint/type standards — several were
written to run on a specific cluster against a specific checkout and are kept because *the numbers
they produced are cited in decisions*. Treat them as the audit trail for those numbers.

## Why it exists

These files previously lived in exactly one place each, none of them version controlled:

* the markdown dossier under `Brainlifts/liv_experiment_research/` in a **non-git** directory;
* ~50 probe scripts on **FarmShare scratch** (`/scratch/users/ericrcwu/liv/`), the only copy.

A machine has died mid-run on this project before. Now there is history.

## Layout

| path | what |
|---|---|
| `research/` | The written record. `00_*`–`08_*` are the original nine-part dossier; `reassessment/` is a from-first-principles re-look; `reassessment/verification/` is the pass that checked the re-look and found four overstated claims. |
| `research/liv-brainlift-experiment-design.md` | The full experiment design (~101 KB). |
| `research/HANDOFF-liv.md` | Snapshot of the repo-root `HANDOFF.md` at the time of import. **The live copy is at the root of `Capstone_LLM` and is the one to trust.** |
| `probes/` | GPU/analysis probes plus their result JSONs, `.sbatch` wrappers, and Slurm logs. |
| `mqar/` | Multi-Query Associative Recall harness — data generator, model, calibration, positive control, and a 43-test suite (`mqar_data_test.py`). |
| `tokenizer_study/` | Vocabulary/fertility work behind the 50,304 decision. |
| `verify/` | Small independent re-derivations (bandwidth, parameter ledger). |
| `logs/` | Assorted run logs kept for provenance. |

Vendored tokenizer vocabularies (`*.tokenizer.json`, ~37 MB of GPT-2/NeoX/OLMo2/Qwen/LFM2 files)
were **deliberately excluded** — they are re-downloadable and dwarfed the actual work. The scripts in
`tokenizer_study/` fetch what they need.

## The probes whose results are load-bearing

| script | what it settled |
|---|---|
| `probes/p1_cache_check.py` | **Reversed an earlier conclusion.** The benchmark that "killed" P1's latency claim held a 40 MiB working set against the L40S's 96 MiB L2, so it measured cache-resident throughput — the one regime where saving bytes buys nothing. Varying *only* residency, with kernel count per step held fixed, flips the sign: −3.7% at 40 MiB → **+39.1%** at 160 MiB → +29.9% at 960 MiB. The real model reads 709 MB/token (7× L2), so past-L2 is the representative rung. Share-weighted to the whole model, ≈**+1.8%** end-to-end. |
| `probes/spectra_v2.py` | Activation-aware gate spectra over 32,768 tokens with `rank(Σ_x)` reported. Keeps P1's premise falsified (gates 493.3 vs a value-stream control at 507.8 — the collapse is a property of the input, not of gates) while showing rank-128 retains **92.6%** of activation-weighted energy. An earlier version used 568 tokens for a 1024×1024 covariance, making `Σ_x` rank-deficient by construction and producing a false positive. |
| `probes/structure_energy.py` | Low-rank r=128 retains **0.929** of activation-weighted energy; grouped g=4 retains **0.130**, identical to a random mask of the same density. Measured on Liquid's *trained* weights, so it is a strong prior about from-scratch training, not a verdict — GaLore is a documented case of this exact proxy failing. |
| `probes/throughput.py` | Measured training throughput for the four pilot arms. See the caveat below. |
| `probes/strip_npy_header.py` | Built the header-free corpus OLMo-core actually needs (see below). |
| `mqar/mqar_calibrate.py` | Calibrated MQAR: vocab **256** (not Zoology's 8192), lr 3e-3, 512k examples. Established the **1/D degenerate floor** — a model that learns "the answer is one of the D values present" without binding scores exactly `1/D`, so the chance baseline moves with the config. |

## Two traps worth reading before reusing any of this

**`np.save` output is not what OLMo-core reads.** `NumpyFSLDataset` reads a *raw, headerless* array:
`_read_chunk_from_array` slices from `index * sequence_length`, and `_get_file_size_and_length`
derives the token count as `file_size // itemsize`. Nothing in the package parses an `.npy` header.
A real `.npy` file carries a 128-byte header = 64 uint16 slots, so every read shifts by 64 tokens —
and because those header bytes decode to ids under 32,032, they index cleanly into a 50,304-row
embedding. **It does not crash.** Training runs, loss falls, everything looks healthy.
`probes/strip_npy_header.py` writes verified header-free copies.

**`TransformerConfig.build()` does not initialize weights.** It constructs modules and leaves
parameter memory uninitialized; `init_weights()` must be called explicitly. The first version of
`probes/throughput.py` omitted it and produced a step-0 loss of ~926 against
`ln(50304) = 10.83`. The throughput numbers were plausible and internally consistent — only a check
on the *magnitude* of the loss caught it. `throughput_results.json` from job 1671574 is that bad run,
kept as the counterexample; it is superseded by the run with `init_weights` and a hard assertion.

Both failures share a shape: **the check that would have caught them tests a value, not the absence
of an exception.**

## Reproducing on FarmShare

```bash
ssh ericrcwu@rice-02.farmshare.stanford.edu        # rice-01 is dead; the login alias points at it
cd /scratch/users/ericrcwu/kda/olmo                # rsync'd checkout, not a git repo
PYTHONNOUSERSITE=1 TRITON_F32_DEFAULT=ieee \
  /scratch/users/ericrcwu/kda/venv/bin/python <script>
```

`PYTHONNOUSERSITE=1` avoids a user-site `torchvision` built against a different torch.
`TRITON_F32_DEFAULT=ieee` is mandatory — the PyTorch tf32 flag does **not** control Triton.

Corpus: `/scratch/users/ericrcwu/liv/data/{train,val}_raw.bin` — 1,200,000,000 GPT-2 FineWeb-Edu
tokens, uint16, headerless, max token id **50,256**. Cluster GPUs are L40S (sm_89, 44.4 GiB
usable); there is no A100 on FarmShare.
