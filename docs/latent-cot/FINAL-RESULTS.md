# Latent-CoT superposition: final results

Definitive numbers from `run_019ffbc1-0a6b-703f-83cf-62fde4175dcf` — all five arms of
`run_019ff929` re-scored at **step 2000** (a matched optimization budget) with the corrected
explicit-CoT answer position. 960 held-out examples, balanced 80/80 reachable/unreachable at each
of depths 2/3/4/5/6/8. Chance = 0.500. Depths 5 and 8 are out of distribution (training used
2/3/4/6).

Report published to
`s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/run_019ffbc1-.../checkpoints/report.json`.

## 1. Per-arm results

| Arm | Mechanism | Accuracy | decodability | D2 | D3 | D4 | D5* | D6 | D8* |
|---|---|---|---|---|---|---|---|---|---|
| **A1** | `no_cot` (floor) | 0.504 | — | .494 | .525 | .506 | .506 | .500 | .494 |
| **A2** | `codi` | 0.586 | 0.592 | **1.000** | .581 | .487 | .531 | .494 | .425 |
| **A3** | `codi` + R1 | 0.596 | **0.990** | **1.000** | .581 | .481 | .619 | .525 | .369 |
| **A4** | `codi` + L2 | **0.653** | 0.644 | **1.000** | .863 | .444 | .562 | .537 | .512 |
| **A0** | `explicit_cot` (ceiling) | **0.952** | — | 1.000 | 1.000 | .988 | .994 | .950 | .781 |

Four things this establishes, none of which was known before this run:

1. **The task is well-posed.** A1 is flat at chance across every depth (.494–.525) and A0 reaches
   0.952. A ~45-point floor-to-ceiling gap means the benchmark can discriminate.
2. **Latent reasoning works, and it is not memorization.** All three CODI arms beat the floor on
   *held-out* data by 8–15 points.
3. **Latent reasoning is shallow.** Every CODI arm is **perfect at depth 2** and at chance by
   depth 4. K = 10 thought slots were available; roughly one to two steps of search are being used.
4. **Explicit CoT is not saturated either.** A0 falls to 0.781 at depth 8, so the hardest
   out-of-distribution case is genuinely hard for every arm.

## 2. Gate A (superposition): negative overall, positive in the deep region

Gate A is `acc_continuous(D) − acc_discrete(D)` = A2 − A0. The theory predicts positive and
increasing with depth.

```
curve: {2: 0.000, 3: -0.419, 4: -0.500, 5: -0.463, 6: -0.456, 8: -0.356}
slope: -0.0394
```

**The headline is negative and the hypothesis is not supported as stated.** Continuous thoughts
equal explicit CoT at depth 2, then fall 42–50 points behind.

But the aggregate slope hides two different regimes, and the decomposition is the interesting part:

| region | slope | what happens |
|---|---|---|
| D2 → D3 | **−0.419** | a cliff: A2 drops 1.000 → 0.581 while A0 stays at 1.000 |
| D4 → D8 | **+0.035** | the gap *narrows*, −0.500 → −0.356 |

In the deep region A0 degrades faster than A2 (−0.207 vs −0.062 from D4 to D8). That is the
*direction* the superposition theory predicts, and it is the first sign of it in the project.

It is weak evidence and should be presented as such: both arms are near chance there
(A2 .487→.425, A0 .988→.781, so A0 is falling from a great height rather than A2 improving), it is
four points from one seed, and the aggregate slope is still negative. The honest statement is that
**the single-number gate A fails, and the shape of the curve suggests the mechanism is
depth-limited rather than depth-averse.**

## 3. Gate B: R1's mechanism succeeded, and that is why it failed

```
A2 (no reg):  acc=0.586   decodability=0.592
A3 (R1):      acc=0.596   decodability=0.990   <-- the finding
A4 (L2):      acc=0.653   decodability=0.644
```

**R1 did exactly what it was designed to do.** `decodability` is the mean top-1 logit-lens
probability of the thoughts, and R1 raised it from 0.592 to **0.990** — it pulled the latent state
onto the vocabulary manifold almost completely. The method is not broken.

**And 0.990 is the problem.** A top-1 probability of 0.99 means each thought decodes to a *single*
token with 99% confidence. The thoughts **collapsed onto individual words**, which is the
destruction of superposition, not its achievement — a superposed state must spread mass over
several candidates.

This was anticipated. From `arms.py`:

> `vocab_reg_entropy_floor` (R1's optional anti-collapse term): it is **off by default** (0.0) so
> the first sweep is the clean A3-vs-A4 comparison, and it lives in the whitelist so it can be
> switched on for A3 alone (empirically, **if the thoughts are seen to collapse**).

**The thoughts were seen to collapse. The pre-registered contingency has fired.** Enabling the
entropy floor for A3 alone is the next experiment; it is already inside the confound whitelist, so
it does not disturb the matched-arm design, and it needs no new code.

Secondary result: both regularized arms beat unregularized A2, and A2 had the *lowest* training CE
(0.167) with the *lowest* held-out accuracy — a clean overfitting signature. Regularization helps
generalization here; the vocabulary-manifold *direction* is not what helps.

## 4. Why the mechanisms differ, conceptually

Both A3 and A4 apply a penalty of identical strength (γ = 0.01) to the thoughts. Only the target
differs:

- **R1 (A3)** decodes each thought through the LM head, forms the *weighted blend* of real token
  embeddings `Σ pᵢ·Eᵢ`, and pulls the thought toward it. A **location** constraint: "be where words
  live". Deliberately a mixture, so superposition is permitted.
- **L2 (A4)** penalizes `thoughts²` — pulls toward the origin, which is not in the vocabulary
  region. A **magnitude** constraint: "be small". No opinion about direction.

A4 is the confound control: if A3 > A4, the gain comes from the vocabulary *direction*; if not,
R1's benefit was generic regularization. It was not.

A geometric side effect worth recording, because it shows in the data: `thought_rms` came out
**A3 1.598 < A4 1.722 < A2 1.789**. R1 shrinks magnitude *harder than the explicit shrinkage
penalty does*, without asking to — because an average of many high-dimensional embeddings partially
cancels, so a mixture target is a shorter vector. So A3 and A4 do not differ purely in
direction-vs-magnitude; A3 does both. The clean separation is **R2** (pull toward the single
nearest token, already implemented and unused): R1-vs-R2 tests "near a word" against "near a
*blend* of words", which is the sharpest available probe of superposition.

## 5. The benchmark was broken first, and finding that is a result

Before any of the above could mean anything: `graph_gen.generate()` gave unreachable targets **zero
in-edges**, making

```
"> T" absent from the edge list   ==   not reachable
```

One substring test solved the task with **1.0000 accuracy at every depth**, out-of-distribution
depths included. The symptom was inverted anchors — the no-CoT *floor* scored a perfect 1.000 while
the explicit-CoT *ceiling* sat at chance. A rule that never consults depth transfers across depth
perfectly, which is what a shortcut looks like from outside.

Repaired with two equal-length layered chains per instance — one rooted at the source, one at an
in-degree-zero decoy root — with the classes differing *only* in which chain the target sits on.
After repair every surface rule tested is at chance:

| Rule | Before | After |
|---|---|---|
| target is an edge destination | **1.0000** | 0.5000 |
| target appears anywhere in edges | **1.0000** | 0.5000 |
| target in-degree > 1 | — | 0.5115 |
| a predecessor of target has a predecessor | — | 0.5000 |
| target id below midpoint | — | 0.5167 |

Guarded by `test_no_surface_heuristic_separates_the_classes`, a battery rather than a single
assertion — because the first repair attempt still leaked at depth 2 and the battery caught it, not
review.

A property that fell out of the redesign: nothing in the construction except the choice of target
consults `reachable`, so **one seed yields the same graph for both labels** with a different node
named as sink. The positive and negative of each pair are exactly matched — a stronger confound
control than the original design had in intent.

## 6. Engineering results

**The CODI memory wall, and why batch size was the wrong lever.** Measured, not argued:

| run | batch | peak at OOM | survived |
|---|---|---|---|
| run_019ff280 | 16 | 38.51 GiB | 2h48m |
| run_019ff806 | 8 | 38.72 GiB | ~2 min |

Halving the batch moved peak memory 0.2%. `arm_loss` sums a per-example loss over the whole batch
and returns one tensor, so nothing is freed until the caller's single `backward()` and every
example's teacher forward *and* K = 10 chain are alive at once. One CODI example is ~2,724
token-forwards against A0's 250 (max 6,158), so peak tracks the *longest examples drawn* — which is
why batch 16 got lucky for three hours on shorter data and batch 8 died immediately on longer.

**Fix: gradient accumulation.** Each batch is sliced and each slice backwarded, scaled by
`len(slice)/len(batch)`, so the accumulated gradient is *identical* to one full-batch backward —
asserted for A0, A2 and A3 by comparing weights after one step from a shared initialization.
**38.72 → 15.6 GiB.** Effective batch size, LR schedule and every confound control untouched.

Also delivered, each from a failure it would have prevented:

- **Wall-clock budget** (`--max-seconds`) so a run killed at the runtime wall still writes
  `metrics.json` and still runs its eval. `metrics.json` is written last, so a killed run used to
  report nothing at all.
- **`--save-every 100`** and **`--match-steps`**: a dense checkpoint ladder plus evaluation at the
  largest step *all* arms reached, so arms that stop at different steps are never compared at
  different optimization budgets.
- **A memory probe that aborts the job** above 34 GiB after two steps, so a wrong memory assumption
  costs two minutes rather than a night.
- **Per-arm `WANDB_RUN_ID`**: all five arms had been resuming into one W&B run and overwriting each
  other, because `wandb.init()` inherits a single platform-set id.
- **A 7-minute heartbeat** naming every arm alive/GONE, because `edullm logs` returns 50 lines and
  the two cheap arms filled them — three dead arms were invisible for two hours.

## 7. The eval bug that nearly buried the result

A0 was measured at **0.518**, essentially chance, while its training CE on the teacher trace had
fallen to **0.001**. A model that writes the BFS trace almost perfectly and then answers at chance
is a broken measurement, not a failed model.

`encode_example` lays the teacher view out as `question <bot> cot <eot> <distill> answer` with the
label mask `True` on the CoT span and `True` on the answer span — but **`False` on both `<eot>` and
`<distill>`**. Nothing ever trains the arm to emit `<distill>`. The eval nevertheless generated
"until `<distill>`" and read the answer logits wherever generation stopped, which for an
unsupervised token means the cap on essentially every example. The answer was also supervised as
`P(answer | … cot <eot> <distill>)`, so reading it without `<distill>` in place is off-distribution.

Fixed by generating until a token the model *is* supervised to emit (`render_cot` ends with `found`
or `none`), then rebuilding `… <eot> <distill>` and reading the answer there — constructing the
position instead of hoping for it.

**A0: 0.518 → 0.952.** Corroborated three ways: A1 held at 0.504 as the untouched control, eval
runtime fell from 27.6 to 9.8 minutes because generation now stops instead of hitting the cap, and
A0's depth curve went from flat noise (.500–.525) to a sensible decline (1.000 → 0.781).

**This one bug was the difference between "no arm can do this task" and a 45-point ceiling.** Every
gate in this document depends on it.

## 8. Caveats to carry with any figure

- **One seed.** Nothing here has a variance estimate.
- **`--lr 2e-5` was never screened**, and it is not even the effective rate: `grad_norm` is
  pre-clip against `max_grad_norm=1.0`, and the arms ran at clip factors of **~21× (A2), ~27× (A3),
  ~4.5× (A4)**. The effective step is roughly 1/20th of nominal.
- **That clipping differs by arm**, so A3's accuracy deficit has a rival explanation in step size,
  not only in R1 being unhelpful. Gate B should not be reported without this.
- **`--batch-size 8`**, below the design's 16, for wall-clock reasons. Identical across arms, so it
  cannot confound arm-vs-arm comparisons; it limits absolute claims only.
- **Depths 5 and 8 are out of distribution** by design, which is why A0's decline concentrates
  there.

## 9. What to do next, in priority order

1. **Turn on `vocab_reg_entropy_floor` for A3.** The pre-registered collapse condition has fired;
   this is the one change with a hypothesis already attached.
2. **Add the R2 arm** (nearest single token). R1-vs-R2 separates "near a word" from "near a blend of
   words" — the direct test of superposition, and the code exists.
3. **Screen the LR**, and consider raising `max_grad_norm`: a 20× clip means none of the arms ran at
   the rate they were nominally given.
4. **A second seed**, before any of these numbers are quoted with confidence.
5. **Supervise `<eot>`/`<distill>` in the label mask.** Deliberately not done here, because it
   changes what every arm trains on and would invalidate comparison against these checkpoints — but
   it is the deeper fix behind §7.

## 10. Run ledger

| Run | Outcome |
|---|---|
| run_019fde30 / 019fde62 / 019df04 | died in seconds — base checkpoint at a bucket that does not exist |
| run_019fecf1 | `FlashAttention only support fp16 and bf16` |
| run_019fee83 | 2h48m, no visible cause — arm logs written to a lost `s3:/` path |
| run_019ff280 | 2h48m, CUDA OOM 38.51 GiB at batch 16; A0/A1 finished, A2–A4 lost |
| run_019ff72e | eval of the above — **found the dataset leak** |
| run_019ff7d9 / 019ff7ed | died in seconds / cancelled — branch lacked philote-dev's bf16 verifier fix |
| run_019ff806 | repaired data; A0/A1 finished, A2–A4 OOM at 38.72 GiB at batch 8 |
| run_019ff95b | re-eval — **A0 0.518 → 0.968**, proving the eval bug |
| run_019ff929 | **all five arms to step 2000** after gradient accumulation |
| **run_019ffbc1** | **final corrected gates — the numbers in this document** |

Nine runs to two gates. Seven were infrastructure; two produced science.
