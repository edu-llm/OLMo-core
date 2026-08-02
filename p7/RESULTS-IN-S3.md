# P7 Result Files Live in S3

On 2026-08-02 the bulk result files under `p7/` moved to S3 and were deleted from this
branch. The code, the configs, the reports, the published figures and the dataset splits
stay in the repository. This document records what moved, where it went and how to get it
back.

If a script in this tree opens a file that does not exist, that file is almost certainly in
S3. Find it with the path mapping below.

## Where the files are

| Repository path | S3 prefix |
| --- | --- |
| `p7/POC/` | `s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts/p7-poc/` |
| `p7/impl3/` | `s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts/p7-impl3/` |

The directory structure below each prefix mirrors the repository exactly, so
`p7/POC/math_eval/math_logic_results_nosi.jsonl` is now
`.../artifacts/p7-poc/math_eval/math_logic_results_nosi.jsonl`. Strip the leading
`p7/POC/` or `p7/impl3/` from the repository path and append the remainder to the matching
prefix.

These are historical ORCD artifacts rather than platform run output, so they sit under
`artifacts/` and have no `run_id`.

The account is 056956104102 and the region is us-east-1. Read access needs the
`sbsandbox` profile or an equivalent role.

## How to get a file back

Fetch a single file.

```bash
export AWS_PROFILE=sbsandbox
aws s3 cp \
  s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts/p7-poc/math_eval/math_logic_results_nosi.jsonl \
  p7/POC/math_eval/
```

Restore a whole directory so a script can run in place.

```bash
export AWS_PROFILE=sbsandbox
aws s3 cp --recursive \
  s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts/p7-poc/curve_run/full_0-923 \
  p7/POC/curve_run/full_0-923
```

Search for a file when the exact path is unknown.

```bash
export AWS_PROFILE=sbsandbox
aws s3 ls --recursive \
  s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts/ | grep judge_out
```

Restored files are not covered by the existing ignore rules in `p7/POC/.gitignore` and
`p7/impl3/.gitignore`, so check `git status` after restoring and do not re-add result files
to the branch.

## What moved

252 files totalling 13.6 MB moved and were deleted. The table lists them by destination
directory.

| S3 directory under `artifacts/` | Files | KB |
| --- | --- | --- |
| `p7-poc/` | 1 | 29 |
| `p7-poc/curve_run/` | 2 | 62 |
| `p7-poc/curve_run/.mplconfig/` | 1 | 125 |
| `p7-poc/curve_run/fine_0-100/grading/` | 28 | 2046 |
| `p7-poc/curve_run/fine_0-100/grading/verify_unique/` | 12 | 236 |
| `p7-poc/curve_run/fine_0-100/judging/` | 24 | 807 |
| `p7-poc/curve_run/full_0-923/grading/` | 45 | 3228 |
| `p7-poc/curve_run/full_0-923/grading/verify_unique/` | 13 | 451 |
| `p7-poc/curve_run/full_0-923/judging/` | 33 | 509 |
| `p7-poc/curve_run/notebooks/` | 2 | 58 |
| `p7-poc/curve_run/raw_data/` | 2 | 763 |
| `p7-poc/curve_run/raw_data/full_0-923_raw/` | 23 | 2934 |
| `p7-poc/day1eval/` | 1 | 264 |
| `p7-poc/general_eval/` | 11 | 160 |
| `p7-poc/kl_analysis/` | 1 | 20 |
| `p7-poc/llm_judge/` | 9 | 58 |
| `p7-poc/math_eval/` | 25 | 1086 |
| `p7-poc/socrateach_sft/` | 2 | 86 |
| `p7-poc/tutor-eval-suite/` | 1 | 52 |
| `p7-impl3/eval/llm_judge/` | 10 | 759 |
| `p7-impl3/eval/llm_judge/t451/` | 6 | 228 |

The moved files fall into these categories.

- Raw generation output under `curve_run/raw_data/`, including the two `curve_out` zips.
- Per checkpoint generation results named `math_logic_results_*.jsonl` and
  `test_results_instruct_*.jsonl`.
- Grading artifacts named `math_logic_graded_*.json`, `needs_verify_*.json`,
  `unique_verify.json`, `tid_to_uid.json` and `uid_verdict.json`.
- Verifier batches and verdicts named `verify_*_batch_*.json`, `batch_*.json` and
  `verifier_out_*.json`.
- Judge batches, keys and verdicts named `judge_input_*.json`, `judge_batch_*.json`,
  `judge_key*.json` and `judge_out_*.json`.
- All 11 `.ipynb` notebooks, which are Colab working copies.
- `curve_run/.mplconfig/fontlist-v390.json`, a matplotlib font cache that was committed by
  accident.

`p7/POC/tutor-eval-suite/` contained only `eval_suite.ipynb`, so that directory no longer
exists in the repository and lives entirely at
`.../artifacts/p7-poc/tutor-eval-suite/eval_suite.ipynb`.

## What stays in the repository

- Every `.py`, `.yaml`, `.sh`, `.sbatch`, `.md` and `.txt` file.
- `p7/impl3/out/ckpt_sweep_bare_hint250.jsonl`. This 194 row file is the verification
  contract that the AWS port is diffed against.
- All 8 `.png` files under `p7/impl3/out/figures/` and all 6 under
  `p7/POC/curve_run/analysis/figures/`. These are the published results and they are small.
- The figure inputs `master_summary.json`, `master_summary_0-100.json` and
  `kl_by_checkpoint.json` under `curve_run/full_0-923/` and `curve_run/fine_0-100/`. The
  scripts in `curve_run/analysis/` read these directly, so the figures stay reproducible
  from the repository alone.
- The small headline aggregates `math_summary.json`, `ped_summary.json` and
  `judge_summary.json`.
- The eval prompt sets `general_prompts.jsonl` and `math_logic_prompts.jsonl`, which are
  inputs rather than results.
- The dataset splits under `p7/POC/ORCD-SFT/data/`, `p7/POC/socrateach_sft/data/` and
  `p7/POC/kl_analysis/colab_uploads/`. These are training inputs. Copies are in S3 for
  durability but the branch keeps them.

## Verification

Every uploaded object was checked before anything was deleted. The upload holds 256
objects totalling 33,333,724 bytes, which is the 252 moved files plus 4 dataset split
files. All 256 objects match their local source on MD5, byte size and key. Five files were
additionally downloaded and compared by SHA-256.

| File | SHA-256 |
| --- | --- |
| `p7/POC/ORCD-SFT/data/socrateach_sft_test.jsonl` | `4812d329135c06589a6b9dabea07e6adb7cca3841bda1ab813bfc82c8850d648` |
| `p7/POC/curve_run/raw_data/curve_out-20260723T222915Z-1-001.zip` | `6adfd9402470b78aedd936f745460c5c21b23655274f4f769ce5010c7be6b16d` |
| `p7/POC/day1eval/colab_eval.ipynb` | `a70b72548311ec45fdb94d3537fe72fe1f540d89c1ea3201411a8cde819947a2` |
| `p7/POC/curve_run/.mplconfig/fontlist-v390.json` | `a30080b9ad3e224b343561b771b152de18110e2a1314422a901a84a48fb65fc3` |
| `p7/impl3/eval/llm_judge/test_results_instruct.jsonl` | `573e04fd95af9a00665167d30d0581f804311e5dc5f7287d1c27a6f1c63b49a0` |

The three copies of `socrateach_sft_test.jsonl` and `socrateach_sft_val.jsonl` under
`ORCD-SFT/data/`, `socrateach_sft/data/` and `kl_analysis/colab_uploads/` are byte
identical, so S3 stores one copy of each under the first two paths rather than three.
