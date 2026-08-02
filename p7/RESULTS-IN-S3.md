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

252 files totalling 13.6 MB moved and were deleted. Two of them were eval inputs that the
retention rule above should have kept, and they have since been restored to the branch, so 250 of
the 252 are absent from the repository today. See the section on the `curve_run` prompt sets below.
The S3 copies of all 252 remain, so the table below describes the upload and is unaffected by that
restore.

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
  inputs rather than results. `math_logic_prompts.jsonl` exists at four paths and every one of
  them stays. Two sit next to the builder that writes them, at `POC/math_eval/` and
  `impl3/eval/math_eval/`, and two sit next to the graders that read them, at
  `POC/curve_run/full_0-923/grading/` and `POC/curve_run/fine_0-100/grading/`. The last two are
  easy to mistake for grading output because of where they live. They are not.
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

The two `curve_run` copies of `math_logic_prompts.jsonl` were later downloaded from S3 and put
back in the branch. Both restored files hash to
`706456465e7fccf564f8157fd1e10b7894707bfe14cb2e5750c9e777b64a0e8d`, which is the hash of the
`POC/math_eval/` copy that never left and of the pre-deletion git blob at `9a7259dd5^`. The
`impl3/eval/math_eval/` copy is a different and larger prompt set at
`ac292e83e7a4593062be84e593656ea74dea120a08ea942e2208d817a46d927d`, so three of the four copies are
byte identical rather than all four.

The three copies of `socrateach_sft_test.jsonl` and `socrateach_sft_val.jsonl` under
`ORCD-SFT/data/`, `socrateach_sft/data/` and `kl_analysis/colab_uploads/` are byte
identical, so S3 stores one copy of each under the first two paths rather than three.

## Scripts that need a restore before they run

Cross referencing every remaining `.py`, `.sh` and `.sbatch` file under `p7/` against the 252
deleted paths finds nine scripts that open a file that is now gone. Each one raises
`FileNotFoundError` before it writes anything, so none of them corrupts an output, but none of
them runs as documented until its inputs come back from S3. Restoring the two `curve_run` prompt
sets did not shorten this list. Those two graders now fail one line later, on the results file
instead of the prompt set, and they fail just as loudly.

| Script | Line | Missing input | Reference kind |
| --- | --- | --- | --- |
| `POC/curve_run/full_0-923/grading/grade_math_logic.py` | 39 | `math_logic_results_<tag>.jsonl` | positional argument |
| `POC/curve_run/fine_0-100/grading/grade_math_logic.py` | 39 | `math_logic_results_<tag>.jsonl` | positional argument |
| `POC/llm_judge/build_batches.py` | 11 | `../test_results_instruct.jsonl` | hardcoded constant, no override |
| `POC/llm_judge/aggregate.py` | 5 | `judge_key.json` | hardcoded constant |
| `POC/general_eval/judge_aggregate.py` | 12, 13 | `judge_key.json`, `general_eval_results.jsonl` | hardcoded constant |
| `POC/general_eval/judge_build.py` | 18 | `general_eval_results.jsonl` | argv default |
| `POC/math_eval/grade_math_logic.py` | 39 | `math_logic_results_<arm>.jsonl` | positional argument |
| `impl3/eval/llm_judge/aggregate.py` | 5 | `judge_key.json` | hardcoded constant |
| `impl3/eval/llm_judge/build_batches.py` | 25 | `test_results_instruct.jsonl`, `t451/test_results.jsonl` | positional argument |

Line numbers are the line whose `open()` raises, as the file stands at HEAD rather than at the
commit that deleted the files. The two `impl3` entries break through the argument the runbook
passes rather than through the argparse default, which points at a path that never existed.

The three `grade_math_logic.py` entries have a second missing input that the table cannot show,
because it is read through a glob rather than an `open()`. Under `--with-verify` each one needs
`verifier_out_<tag>_*.json`, and those verdicts are results that stay in S3. The graders now refuse
to run without them, so restore the whole grading directory rather than the results file alone.

### Restore commands

Run these from the repository root. Each block restores everything the named scripts read.

```bash
export AWS_PROFILE=sbsandbox
S3=s3://sbsandbox-intern-edullm-outputs/teams/post-training/artifacts

# POC/curve_run/full_0-923/grading/grade_math_logic.py
aws s3 cp --recursive $S3/p7-poc/curve_run/full_0-923/grading \
  p7/POC/curve_run/full_0-923/grading

# POC/curve_run/fine_0-100/grading/grade_math_logic.py
aws s3 cp --recursive $S3/p7-poc/curve_run/fine_0-100/grading \
  p7/POC/curve_run/fine_0-100/grading

# POC/llm_judge/build_batches.py and POC/llm_judge/aggregate.py
aws s3 cp $S3/p7-poc/test_results_instruct.jsonl p7/POC/
aws s3 cp --recursive $S3/p7-poc/llm_judge p7/POC/llm_judge

# POC/general_eval/judge_build.py and POC/general_eval/judge_aggregate.py
aws s3 cp --recursive $S3/p7-poc/general_eval p7/POC/general_eval

# POC/math_eval/grade_math_logic.py
aws s3 cp --recursive $S3/p7-poc/math_eval p7/POC/math_eval

# impl3/eval/llm_judge/build_batches.py and impl3/eval/llm_judge/aggregate.py
aws s3 cp --recursive $S3/p7-impl3/eval/llm_judge p7/impl3/eval/llm_judge
```

Every block above is recursive on purpose. Fetching one `math_logic_results_<tag>.jsonl` on its own
leaves `verifier_out_<tag>_*.json` behind, and the graders now refuse that rather than grading it,
so a single file restore costs a round trip. The two `curve_run` blocks re-download
`math_logic_prompts.jsonl` over the copy the branch already carries. That is harmless because the
S3 object and the tracked file are the same bytes, and `git status` will show no change.

### The silent failure is now impossible

`grade_math_logic.py` used to under-report rather than fail when its verifier output was missing.
Every MATH-500 answer that needs symbolic equivalence checking is scored wrong until a verdict in
`verifier_out_<tag>_*.json` says otherwise, so a `--with-verify` run whose verdicts had gone to S3
matched zero files, quietly scored those answers wrong, printed `grading complete`, overwrote
`math_logic_graded_<tag>.json` with the wrong rows and exited 0. On the `nosi` arm of
`POC/math_eval/` that moved base MATH-500 from 3/25 to 2/25 and the overall figure from 13/70 to
12/70. On the `c300` arm of `curve_run/full_0-923/` it moved the SFT column from 6/70 to 4/70,
while the `base` arm of the same directory happened to be unaffected, which is what made the bug
hard to see by inspection.

All three copies of the script now carry two guards that fire only under `--with-verify`. The
first refuses when the verifier glob matches nothing. The second refuses when the glob matches but
some answer still has no verdict, which is the same under-count reached through a partial restore.
Both exit non-zero before printing a table or writing `math_logic_graded_<tag>.json`, so a bad
restore can no longer clobber a good graded file. The stage 1 pass without `--with-verify` is
untouched and still writes `needs_verify_<tag>.json`.

Every archived arm satisfies both guards. All 20 result files across `POC/math_eval/`,
`curve_run/full_0-923/grading/` and `curve_run/fine_0-100/grading/` were regraded from a full S3
restore and each reproduced its archived `math_logic_graded_<tag>.json` byte for byte.

The three copies of `grade_math_logic.py` were byte identical before this change and are byte
identical after it. Keep them that way, and check it after any edit rather than trusting a hash
written down here, which would go stale on the first legitimate change.

```bash
md5 -q p7/POC/math_eval/grade_math_logic.py \
  p7/POC/curve_run/full_0-923/grading/grade_math_logic.py \
  p7/POC/curve_run/fine_0-100/grading/grade_math_logic.py | sort -u
```

That must print exactly one line. If a guard is edited in one copy and not the others, the arm
that runs from the unpatched copy is the one that goes back to reporting a number quietly too low,
and the divergence is invisible in a diff of the commit that caused it.

### The two curve_run prompt sets came back

`math_logic_prompts.jsonl` is an eval input, and the retention rule above keeps it. That rule
reached `POC/math_eval/` and `impl3/eval/math_eval/` but missed the copies under
`curve_run/full_0-923/grading/` and `curve_run/fine_0-100/grading/`, which `9a7259dd5` deleted.
Both have been restored from S3 and are back in the branch, verified against the retained
`POC/math_eval/` copy by SHA-256 as recorded in the verification section above.

They were deleted because they sit in a directory named `grading` alongside genuine grading
output. Filename is not the test and neither is the parent directory. The test is whether the
grader reads the file or writes it, and `grade_math_logic.py` reads this one on line 38. Do not
delete these two again. If a future migration sweeps `curve_run/*/grading/`, exclude
`math_logic_prompts.jsonl` explicitly.

The S3 objects at both paths were left in place. They cost 2 objects, they are a durable copy of a
file the branch is now proven to match, and deleting them would break the recursive restore
commands above.

### What still works

The six figure scripts under `POC/curve_run/analysis/` read only `master_summary.json` and
`master_summary_0-100.json`, resolve them relative to `__file__`, and are unaffected.
`impl3/eval/make_figures.sh` and `impl3/eval/plot_figure3.py` default to
`out/ckpt_sweep_bare_hint250.jsonl`, which stays. Every published figure therefore remains
reproducible from the branch alone.

The stale defaults in `impl3/eval/plot_figure3.py` and `impl3/eval/sweep_ckpt_eval.py` that name
`out/ckpt_sweep_eval.jsonl` predate this migration and are not caused by it. The same applies to
`data/socrateach_sft_val.jsonl` throughout `impl3`, which `snapshot_hf_dataset.py` generates at
run time and which the branch never carried.
