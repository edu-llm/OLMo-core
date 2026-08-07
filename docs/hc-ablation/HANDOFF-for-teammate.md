# mHC / Hyper-Connections: handoff

**From:** Adarsh Rajesh
**Covers:** Jul 19 - Aug 7, 2026
**Read this if:** you are picking up the question of whether OLMo-core should adopt Manifold-Constrained Hyper-Connections.

---

## Orientation (read first)

Hyper-Connections (HC) replace the single residual stream with `n` parallel streams. Each
sub-layer reads one vector in from those streams, runs unchanged attention or MLP, writes the
result back, and a matrix `H_res` mixes the streams between layers. mHC constrains `H_res` to be
doubly stochastic so repeated mixing over depth cannot amplify or suppress the signal.

Adopting it means changing the tensor contract between transformer blocks from `(B, T, D)` to
`(B, T, n, D)`. That is the whole cost story.

**Bottom line: don't merge into main yet.** The published gain is inside seed noise. There is one
open question that is cheap to answer and would be genuinely new, described at the bottom.

**A working prototype exists.** Six-arm ablation harness, correctness gate passing, zero
regressions. No training and no evaluation numbers.

### Blocker for you specifically

**The branch is local on Adarsh's machine and has not been pushed.** You cannot fetch it yet. Ask
him to push `edullm/adarsh-hc-ablation` before you try to run anything below.

---

## Current state

| | |
| --- | --- |
| Branch | `edullm/adarsh-hc-ablation`, 4 commits, **not pushed** |
| Base | `origin/main` at `08df5aa0` |
| Commits | `9849771d` core module, `c0a1a149` ablation + gate + docs, `b8939e3a` RFC, `9967c357` handoff |
| Correctness gate | 30/30 pass (35/35 with `--check-compile`) |
| Test suite | 525 pass, 0 fail, 0 regressions |
| Training / eval numbers | **None.** Needs a GPU. |

### Files

Library (what would actually ship, 1,221 insertions, **2 deletions**):

- `src/olmo_core/nn/hyper_connections.py` (787) - the five mixer parameterizations, Sinkhorn projection, config with parameter-count arithmetic, read-in / mix / write-out forward path, identity-preserving init, symmetry breaking, stream-collapse readout
- `src/olmo_core/nn/transformer/hc_block.py` (176) - HC-wrapped block matching OLMo-2's `z + LN(f(z))` ordering
- `src/olmo_core/nn/transformer/model.py` (+140) - stream expand after embeddings, collapse before `lm_head`
- `src/olmo_core/nn/transformer/config.py` (+110), `__init__.py` (+10, -2)

Supporting (2,004 lines): `src/scripts/ablations/hc_ablation.py` (556),
`src/test/nn/hyper_connections_test.py` (515), `src/scripts/ablations/hc_gate1_check.py` (463),
`docs/hc-ablation/` (388), `src/test/scripts/hc_ablation_test.py` (81).

**The two deletions are the number that matters.** They are an import-line edit. Nothing existing
is rewritten and the single-stream path is untouched, which is why all 525 existing tests pass. Of
`hyper_connections.py`'s 788 lines only ~286 are executable code; 345 are Sphinx docstrings the
repo requires. The entire mechanism is roughly 370 lines of real logic.

### How to run it

Everything below is CPU-only. No GPU, no AWS, no `edullm` command.

```bash
# The repo's own .venv is x86_64 under Rosetta, capped at torch 2.2.2, and olmo_core
# will not import there. Use an arm64 Python 3.12 venv with a current torch.
python src/scripts/ablations/hc_gate1_check.py     # correctness gate, exits nonzero on failure
python src/scripts/ablations/hc_ablation.py --dry-run   # builds all six arms, prints param table
pytest -v src/test/nn/hyper_connections_test.py
```

`DATA_PATHS` in the ablation script is deliberately empty. Pointing the arms at a corpus is a
decision for whoever runs them.

### The six arms

Measured parameter counts at `n=4` on the 190M config (267,424,512 total params). These reproduce
the papers' arithmetic exactly, which is itself a correctness check.

| Arm | Mixer | Routing params | Per sub-layer | Formula |
| --- | --- | --- | --- | --- |
| `baseline` | none, 1 stream | 0 | 0 | - |
| `mhc_identity` | fixed `I` | 192 | 8 | `2n` |
| `kromhc` | Kronecker | 288 | 12 | `2log₂n + 2n` |
| `hc_unconstrained` | raw | 576 | 24 | `n² + 2n` |
| `mhc_sinkhorn` | Sinkhorn | 576 | 24 | `n² + 2n` |
| `mhc_lite` | Birkhoff | 768 | 32 | `n! + 2n` |

Evals are wired to OLMo-core's native task names: ARC easy/challenge, HellaSwag, PIQA, Winogrande
for classic; GSM8K and the seven Minerva-MATH subtasks for reasoning and math. These need the
`eval` extra (`ai2-olmo-eval==0.9.0`) and a GPU.

---

## Not done, and do not claim otherwise

- No training. No evaluation numbers. Nothing has been run on a GPU.
- Static routing only. Dynamic routing, MoE integration and fused kernels are unimplemented, per
  the internal implementation order that says stabilize the static dense reference first.
- Tensor and context parallelism **raise `NotImplementedError`** on a hyper-connected block rather
  than silently applying a plan written for a 3-D hidden state.
- FSDP is inherited and untested against a 4-D stream tensor.
- Compile parity is verified for a block's forward pass on CPU only. The ablation leaves
  `compile_model=False`.
- Gate 2 (science) and Gate 3 (systems) have not started.

---

## The question log

What was asked, what came back, and why it was asked. Prompts are verbatim, grammar corrected only.

### Jul 19 - is this a live research line?

**Q:** For each section, give a human-readable summary and the main takeaway. Then use 50 agents to find the top 5 papers in that topic from the last 3 months, a 50-word summary each, and a takeaway on how it might change the conclusion.
**A:** Surveyed the derivative literature; mHC is actively contested.
**Why:** Deciding whether mHC was worth investing in before reading deeply.
**Found:** mHC-lite and KromHC (Jan 2026), Ablate and Rescue (Mar), go-mHC (Apr), Stream Collapse (2606.03483, Jun, one stream dominates), Expanded Hyper-Connections (2607.14530, Jul). mHC amplification ~1.6 vs ~3000 for unconstrained HC.

### Early Aug - scoping the proposal

**Q:** What I have expertise in is mHC. I spent around 8 hours reading that paper 2 weeks ago, and other papers around it for 2 hours.
**A:** Recommended mHC over MLA; the reading depth is the differentiator.

**Q:** Does OLMo already have an implementation of mHC? Also, the guy leading us, Ben, is really big on MoE. He said Kimi is too complex and DeepSeek is a better focus, and we have a 280 billion token dataset we might expand as the model trains.
**A:** No mHC anywhere in the repo. mHC composes with MoE rather than competing with it.
**Why:** Checking two possible disqualifiers: redundancy, and fit with the team's stated direction.

**Q:** Can you even do experiments by replacing mHC only at inference time and not using it in training?
**A:** Inference-only shows nothing. Frozen-backbone PEFT trains 772 params, laptop-scale.
**Why:** Looking for an experiment small enough to actually run.

**Q:** How many lines of code in addition would it take, and what do those lines of code mean in the file?
**A:** 1,221 library lines, only 2 deletions. Almost purely additive. Breakdown in "Files" above.
**Why:** Sizing review burden and blast radius.

**Q:** Investigate each of these papers and tell me if I should have made line edits, added a citation, or done something. [50 papers]
**A:** Four verified. One major error, one missing citation.
**Found:** The xHC label was inverted; the stream-collapse claim cited the wrong paper (correct one is 2606.03483). Both fixed.

**Q:** Pull up what different people have seen, and in what circumstances: "Is this a good idea, e.g. performance up? Yes, author-reported, not replicated."
**A:** "Not replicated" is wrong. Two independent groups reproduced mHC beating baseline.

**Q:** Write parameter count, token count, pretrained or post-trained, similarity to OLMo, and percent improvement over non-mHC or HC, for the original mHC paper plus two others.
**A:** Three papers tabulated. Two percentages are derived arithmetic, not printed in source.
**Warning:** The 500B-token figure is DHC, not mHC. Do not cite it as mHC.

### Aug 6-7 - the PEFT paper and the decision

**Q:** Summarize this fast. [arXiv 2607.18130, Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning, Jul 20 2026]
**A:** mHC works but loses to LoRA; identity mixing wins.

**Q:** Did they implement mHC on the OLMo architecture?
**A:** Yes, wrapped frozen OLMo-2 1B and 7B.

**Q:** Is mHC an inference-type thing, or also in pretraining or post-training?
**A:** Architecture mechanism; trained during pretraining or post-training, executes at inference.

**Q:** So this paper post-trained an OLMo architecture on mHC and found it did worse than a single-stream residual?
**A:** Partly wrong, and the correction matters. mHC beat the frozen model and lost to LoRA. Multi-stream still won; only cross-stream *mixing* lost.

**Q:** Does OLMo-core use only one residual stream or multiple?
**A:** One. Multiple streams are an mHC addition.

**Q:** What number of n streams is most useful?
**A:** n=4. Past it, loss barely moves and memory scales linearly.
**Found:** n=4 is the field default in both the HC and mHC papers. Loss improves monotonically with sharp diminishing returns: dynamic KromHC on OLMo-2-1B at 20k steps gives test loss 1.37 (n=2), 1.35 (n=4), 1.33 (n=8), 1.32 (n=16), so the last step buys 0.01 while trainable params go 12.1M to 43.0M. Downstream does not track loss at all: n=2 is best on PIQA, n=4 on TriviaQA, n=8 on HellaSwag and MMLU. With `H_res` fixed to identity the large-n benefit grows (n=16 drops 1.32 to 1.27 while saving 8.39M params). Cost is linear in n. KromHC requires n a power of 2. Every one of these numbers is frozen-backbone finetuning; no pretraining sweep over n exists.

**Q:** Why is that number of n streams the most useful?
**A:** The gain lives in the read/write gates, which saturate. The costs don't.
**Found:** Three reasons converge.

1. The benefit is not where the name suggests. The component ablation (dynamic KromHC, 3000 steps) shows removing `h_pre` costs +0.14 train loss and removing `h_post` costs +0.12, while removing the cross-stream mixer `H_res` costs only +0.05. Removing both gates collapses to baseline (+0.64). Interpretability agrees: `H_res` drifts back toward identity in later layers, so the model actively suppresses mixing. Read/write gates are a convex combination over n streams, and once there are enough for a few distinct read patterns, more are redundant. That is the curve's shape: -0.02, -0.02, -0.01.
2. Cost grows faster than benefit. Activation memory is linear in n. Dynamic routing params roughly quadruple per doubling of n (918K, 3.41M, 12.1M, 43.0M for n=2,4,8,16).
3. **Specific to this ablation:** n=4 is the only value where all three mixer parameterizations stay comparable. Mixer cost is n² for Sinkhorn, n! for mHC-lite, 2log₂n for KromHC. At n=4 that is 16, 24 and 4. At n=8 it becomes 64, **40320** and 6, and mHC-lite stops being a fair arm. Any n above 4 silently turns a three-way comparison into a two-way one.

Separately, the stream-collapse work (2606.03483) finds one stream tends to dominate, so extra streams may be capacity the model never uses.

**Q:** What are the other main variants of HC, like xHC or something else, when did they come out, and why?
**A:** Eleven papers in 22 months, in four reaction waves. Dates from arXiv IDs (YYMM).

*Foundations.* HC (Zhu et al., 2409.19606, Sep 2024): n parallel streams, learned read/mix/write, static and dynamic. Unconstrained mixing compounds over depth, amplification near 3000. mHC (Xie et al., 2512.24880, Dec 2025, DeepSeek): constrains the mixer to doubly stochastic via Sinkhorn-Knopp, amplification ~1.6. Everything else reacts to mHC.

*Wave 1, cheaper routes to the same constraint.* All target the 20 Sinkhorn iterations. mHC-lite (Yang & Gao, 2601.05732, Jan 2026), Birkhoff-von Neumann, exact and iteration-free but useless past small n. KromHC (Zhou et al., ICML 2026), Kronecker product of 2x2 doubly stochastic factors, cheapest, needs n a power of 2. go-mHC (Dandachi & Diggs-Galligan, 2604.02309, Apr 2026), generalized orthostochastic matrices. Accelerating Birkhoff Projection (Wang & Qiu, 2606.07574, Jun 2026), speeds the projection rather than replacing it.

*Wave 2, arguing the constraint is wrong.* JPmHC (Sengupta et al., 2602.18308, Feb 2026), orthogonal HC for dynamical isometry. Spectral-sphere-constrained HC (Liu et al., 2603.20896, Mar 2026), beyond the Birkhoff polytope. TBP-mHC (Lyubinin, 2605.21724, May 2026), transportation polytopes.

*Wave 3, diagnosis rather than architecture.* Ablate and Rescue (Peng et al., 2603.14833, Mar 2026), some streams dominate causally while similar ones sit passive. Analyzing Stream Collapse (Alimaskina et al., 2606.03483, Jun 2026), one dominant stream, deep-layer mixing collapsing to identity.

*Wave 4, scaling n past 4.* xHC, Expanded Hyper-Connections (Zhang et al., 2607.14530, Jul 16 2026). Also Sparse Selective HC (Modi et al., SoutheastCon 2026).

**Important qualification.** xHC verifies the "why n=4" reasoning and then partly overturns the
conclusion. It names the same two bottlenecks, write-back information not scaling and dense
residual-mixing generation costing O(N³C), and shows both are fixable. With temporal feature
augmentation plus a sparse architecture updating only k=4 of N=16 streams while reading all 16, it
reports average downstream 40.6 vanilla, 44.8 mHC, 48.8 xHC on an 18B MoE, and 47.8, 50.5, 53.6 on
28B. Naive mHC at N=16 moved loss only 2.004 to 1.998 for +18.8% FLOPs; xHC reached 1.983 for
+3.3%. So n=4 is still right for a plain mHC implementation and for this ablation, but "n=4 is the
ceiling" stopped being true in mid-July 2026. **xHC's gains are measured on MoE pretraining**,
which is the setting this team actually cares about.

**Q:** Build an ablation test for all 3 primary variations of HC at n=4 with OLMo-core.
**A:** Six-arm harness built; correctness gate passes 30/30. Details in "Current state" above.

**Q:** Give me 200-300 words of real analysis on fully implementing this into OLMo-core and where the expected performance gains are.
**A:** The gains sit inside the noise floor.

**Q:** This is for the actual main branch, a plan my team wants before they decide on the implementation.
**A:** Written as a go/no-go RFC: `docs/hc-ablation/RFC-adopting-mhc.md`, ~1,050 words, commit `b8939e3a`. Recommends not merging yet.

---

## Recommendation

**Do not merge mHC into `main` now.** The published gain is 0.980 against 0.981 test loss at 7B,
with downstream wins splitting four benchmarks to four, on limited seed coverage. That is inside
ordinary seed-to-seed variation. It does not justify a high-risk change to a core tensor contract.

The real cost is not parameters (192 to 768) but 4x residual activation memory and rewriting the
block tensor contract, which pulls in pipeline parallelism (very high risk), expert parallelism
(high), and a defined load path for every published OLMo checkpoint (high). **It also collides
with live work:** mHC has to touch the MoE blocks that hand-code their residual adds, and those are
under active experimentation on `edullm/moe-m1-pilot`.

**Do fund Gate 2 at small scale.** The one genuinely open question: the finetuning paper found that
fixing `H_res` to the identity, meaning not mixing streams at all, matched or beat learned mixing.
That contradicts the original mHC paper, where learned mixing is the point. **Nobody has tested
which holds during pretraining, and OLMo-core is a pretraining library.**

If identity wins, we skip the expensive learned-mixing machinery entirely. If mixing wins, we have
our own evidence instead of someone else's and Gate 3 becomes worth funding. Either outcome is
informative, which is what makes it the right next spend.

**Gate 2 requires multiple seeds per arm with reported variance.** At these effect sizes a single
run per arm cannot distinguish a real improvement from luck. This is not optional rigour; without
it the experiment cannot answer the question it is being run to answer.

The harness already exists and passes Gate 1. What is missing is a corpus, a GPU, and seeds.
