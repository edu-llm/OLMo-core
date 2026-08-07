# Agent Handoff: mHC / Hyper-Connections

Owner: Adarsh Rajesh. Span: Jul 19 - Aug 7, 2026. Four chats.
Useful probes only. Formatting, grammar-fix and edit-application prompts removed.
Prompts are verbatim, grammar corrected only.

---

## Part 1 - Jul 19: is this a live research line?

**PROBE:** For each section, give a human-readable summary and the main takeaway. Then use 50 agents to find the top 5 papers in that topic from the last 3 months, a 50-word summary each, and a takeaway on how it might change the conclusion.
**ANSWER:** Surveyed the derivative literature; mHC is actively contested.
**DIRECTION:** Deciding whether mHC was worth investing in before reading deeply.
**FINDINGS:** mHC-lite and KromHC (Jan 2026), Ablate and Rescue (Mar), go-mHC (Apr), Stream Collapse (2606.03483, Jun, one stream dominates), Expanded Hyper-Connections (2607.14530, Jul, scales streams past 4). mHC amplification ~1.6 vs ~3000 for unconstrained HC. HyperDiT excluded as unrelated.

---

## Part 2 - early Aug: the proposal to the team

**PROBE:** What I have expertise in is mHC. I spent around 8 hours reading that paper 2 weeks ago, and other papers around it for 2 hours.
**ANSWER:** Recommended mHC over MLA; 10 hours of reading is the differentiator.
**DIRECTION:** Testing whether prior reading depth justifies picking mHC as the proposal topic.

**PROBE:** Does OLMo already have an implementation of mHC? Also, the guy leading us, Ben, is really big on MoE. He said Kimi is too complex and DeepSeek is a better focus, and we have a 280 billion token dataset we might expand as the model trains.
**ANSWER:** No mHC anywhere in the repo. mHC composes with MoE rather than competing.
**DIRECTION:** Checking the two disqualifiers named in the call: redundancy, and the lead's stated preference.

**PROBE:** Can you even do experiments by replacing mHC only at inference time and not using it in training?
**ANSWER:** Inference-only shows nothing. Frozen-backbone PEFT trains 772 params, laptop-scale.
**DIRECTION:** Finding an experiment small enough to actually run before the deadline.

**PROBE:** How many lines of code in addition would it take, and what do those lines of code mean in the file?
**ANSWER:** 1,221 library lines, only 2 deletions. Almost purely additive.
**DIRECTION:** Sizing the change so the team can judge review burden and blast radius.
**FINDINGS:** Now measured from the working implementation rather than estimated. The library change that would actually ship is 1,221 insertions and 2 deletions across 5 files: `nn/hyper_connections.py` (787 new), `nn/transformer/hc_block.py` (176 new), `nn/transformer/model.py` (+140), `nn/transformer/config.py` (+110), `nn/transformer/__init__.py` (+10, -2). Supporting material is another 2,004 lines: the ablation runner (556), unit tests (515), the GATE_1 correctness script (463), docs (388), and a script test (81). Total 3,225 lines across 14 files.

What those lines actually are matters more than the count. Of `hyper_connections.py`'s 788 lines, only about 286 are executable code; 345 are docstrings the repo requires for Sphinx, 138 blank, 19 comments. `hc_block.py` is 83 lines of code out of 177. So the entire mechanism is roughly 370 lines of real logic: the five mixer parameterizations and the Sinkhorn projection, the config dataclass with its parameter-count arithmetic, the read-in / mix / write-out forward path, identity-preserving initialization and symmetry breaking, the stream-collapse readout, and a block that wraps attention and feed-forward the way `ReorderedNormTransformerBlock` does.

**The two deletions are the number to quote.** They are an import line edit. Nothing existing is rewritten, and the single-stream path is untouched, which is why all 525 existing tests still pass.

**PROBE:** Investigate each of these papers and tell me if I should have made line edits, added a citation, or done something. [50 papers]
**ANSWER:** Four verified. One major error, one missing citation.
**DIRECTION:** Systematic sweep for citation errors across the HC literature.
**FINDINGS:** xHC label was inverted; the stream-collapse claim cited the wrong paper (correct one is 2606.03483). Both fixed with two-line edits.

**PROBE:** Pull up what different people have seen, and in what circumstances: "Is this a good idea, e.g. performance up? Yes, author-reported, not replicated."
**ANSWER:** "Not replicated" is wrong. Two independent groups reproduced mHC beating baseline.
**DIRECTION:** Stress-testing the weakest claim in the draft.

**PROBE:** Write parameter count, token count, pretrained or post-trained, similarity to OLMo, and percent improvement over non-mHC or HC, for the original mHC paper plus two others.
**ANSWER:** Three papers tabulated. Two percentages are derived arithmetic, not printed in source.
**DIRECTION:** Building a defensible evidence table.
**NOTE:** The 500B-token figure is DHC, not mHC. Do not cite it as mHC.

---

## Part 3 - Aug 6-7: the PEFT paper and the OLMo-core decision

**PROBE:** Summarize this fast. [arXiv 2607.18130, Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning, released Jul 20 2026]
**ANSWER:** mHC works but loses to LoRA; identity mixing wins.
**DIRECTION:** Triaging a new paper for relevance before spending real time on it.

**PROBE:** Did they implement mHC on the OLMo architecture?
**ANSWER:** Yes, wrapped frozen OLMo-2 1B and 7B.
**DIRECTION:** Confirming the paper is directly applicable to his own codebase.

**PROBE:** Is mHC an inference-type thing, or also in pretraining or post-training?
**ANSWER:** Architecture mechanism; trained in pretraining or post-training, executes at inference.
**DIRECTION:** Locating which stage of the pipeline mHC would actually touch.

**PROBE:** So this paper post-trained an OLMo architecture on mHC and found it did worse than a single-stream residual?
**ANSWER:** Partly wrong. Better than frozen, worse than LoRA; only cross-stream mixing lost.
**DIRECTION:** Testing his own reading. The correction matters: multi-stream still won, mixing did not.

**PROBE:** Does OLMo-core use only one residual stream or multiple?
**ANSWER:** One. Multiple streams are an mHC addition.
**DIRECTION:** Pinning the exact baseline he would be modifying.

**PROBE:** What number of n streams is most useful?
**ANSWER:** n=4. Past it, loss barely moves and memory scales linearly.
**DIRECTION:** Choosing the expansion rate before spending compute on the ablation.
**FINDINGS:** n=4 is the field default in both the HC and mHC papers. Loss improves monotonically with sharp diminishing returns: dynamic KromHC on OLMo-2-1B at 20k steps gives test loss 1.37 (n=2), 1.35 (n=4), 1.33 (n=8), 1.32 (n=16), so the last step buys 0.01 while trainable params go 12.1M to 43.0M. Downstream does not track loss at all: n=2 is best on PIQA, n=4 on TriviaQA, n=8 on HellaSwag and MMLU. With `H_res` fixed to identity the large-n benefit grows (n=16 drops 1.32 to 1.27 while saving 8.39M params). Cost is linear in n, since the model carries n copies of the residual state between every layer, which binds harder in pretraining than finetuning. KromHC additionally requires n to be a power of 2. Caveat: every one of these numbers is frozen-backbone finetuning. No pretraining sweep over n exists.

**PROBE:** Why is that number of n streams the most useful?
**ANSWER:** The gain lives in read/write gates, which saturate; costs don't.
**DIRECTION:** Wanting the mechanism behind n=4, not just the empirical curve, so the choice survives questioning.
**FINDINGS:** Three reasons converge.

First, the benefit does not come from where the name suggests. The component ablation (dynamic KromHC, 3000 steps) shows removing `h_pre` costs +0.14 train loss and removing `h_post` costs +0.12, while removing the cross-stream mixer `H_res` costs only +0.05. Removing both gates collapses it to baseline (+0.64). The value is in how a layer reads from and writes to the streams, not in mixing them. Interpretability agrees: `H_res` drifts back toward identity in later layers, so the model actively suppresses mixing. Read/write gates are a convex combination over n streams, and once there are enough streams for a few distinct read patterns, more are redundant. That is the shape the loss curve shows: -0.02, -0.02, then -0.01.

Second, cost grows faster than benefit. Activation memory is linear in n. Dynamic routing parameters roughly quadruple per doubling of n (918K, 3.41M, 12.1M, 43.0M for n=2,4,8,16). Paying 4x memory and 3.5x parameters for 0.01 loss is not a trade worth taking.

Third, and specific to this ablation: n=4 is the only value where all three mixer parameterizations stay comparable. Mixer cost is n² for Sinkhorn, n! for mHC-lite, 2log₂n for KromHC. At n=4 that is 16, 24 and 4, the same order of magnitude, so the arms are genuinely matched. At n=8 it becomes 64, 40320 and 6, and mHC-lite stops being a fair arm. Any n above 4 silently turns a three-way comparison into a two-way one.

Separately, the stream-collapse work (2606.03483) finds one stream tends to dominate, so extra streams are capacity the model never uses.

**PROBE:** What are the other main variants of HC, like xHC or something else, when did they come out, and why?
**ANSWER:** Eleven papers in 22 months, in four distinct reaction waves.
**DIRECTION:** Mapping the field so the proposal cites the right work and isn't blindsided.
**FINDINGS:** Dates from arXiv identifiers (YYMM). Grouped by the problem each reacts to.

*The two foundations.* HC (Zhu et al., 2409.19606, Sep 2024) expanded the residual stream into n parallel streams with learned read, mix and write, in static and dynamic forms. Unconstrained mixing compounds over depth, with measured amplification near 3000. mHC (Xie et al., 2512.24880, Dec 2025, DeepSeek) constrains the mixer to doubly stochastic via Sinkhorn-Knopp, bringing amplification to about 1.6. Everything below reacts to mHC.

*Wave one, cheaper ways to get the same constraint.* All target the 20 Sinkhorn iterations. mHC-lite (Yang & Gao, 2601.05732, Jan 2026) uses Birkhoff-von Neumann, exact and iteration-free but useless past small n. KromHC (Zhou et al., ICML 2026) uses a Kronecker product of 2x2 doubly stochastic factors, only 2log₂n parameters, but needs n a power of 2. go-mHC (Dandachi & Diggs-Galligan, 2604.02309, Apr 2026) parameterizes via generalized orthostochastic matrices. Accelerating Birkhoff Projection (Wang & Qiu, 2606.07574, Jun 2026) speeds up the projection instead of replacing it.

*Wave two, arguing the constraint itself is wrong.* JPmHC (Sengupta et al., 2602.18308, Feb 2026) uses orthogonal hyper-connections for dynamical isometry. Spectral-sphere-constrained HC (Liu et al., 2603.20896, Mar 2026) goes explicitly beyond the Birkhoff polytope. TBP-mHC (Lyubinin, 2605.21724, May 2026) uses transportation polytopes to recover expressivity the constraint gives up.

*Wave three, diagnosis rather than new architecture.* Ablate and Rescue (Peng et al., 2603.14833, Mar 2026) finds some streams dominate causally while representationally similar ones sit passive. Analyzing Stream Collapse (Alimaskina et al., 2606.03483, Jun 2026) finds one dominant stream and deep-layer mixing collapsing to identity. This wave questions whether the extra streams are used at all.

*Wave four, scaling n past 4.* xHC, Expanded Hyper-Connections (Zhang et al., 2607.14530, Jul 16 2026). Also Sparse Selective HC (Modi et al., SoutheastCon 2026).

**Qualification to the previous probe.** xHC verifies the "why n=4" reasoning and then partly overturns the conclusion. It names the same two bottlenecks, write-back information not scaling and dense residual-mixing generation costing O(N³C), and shows both are fixable. With temporal feature augmentation plus a sparse architecture updating only k=4 of N=16 streams while reading all 16, it reports average downstream 40.6 vanilla, 44.8 mHC, 48.8 xHC on an 18B MoE, and 47.8, 50.5, 53.6 on 28B. Naive mHC at N=16 moved loss only 2.004 to 1.998 for +18.8% FLOPs; xHC reached 1.983 for +3.3%. So n=4 remains correct for a plain mHC implementation and for this ablation, but "n=4 is the ceiling" stopped being true in mid-July 2026. Worth flagging to Ben: xHC's gains are measured on MoE pretraining, exactly his setting.

**PROBE:** Build an ablation test for all 3 primary variations of HC at n=4 with OLMo-core. Keep it on the branch, don't push to main or disturb anyone else.
**ANSWER:** Built six-arm harness on a branch; correctness gate passes 30/30.
**DIRECTION:** Moving from reading to an instrument that can settle the question himself.
**DELIVERED:** Branch `edullm/adarsh-hc-ablation`, local only, unpushed. Five mixers (identity, unconstrained, sinkhorn, birkhoff, kronecker) plus single-stream baseline. Routing params 192/288/576/576/768 on the 190M config, matching the paper's n²+2n, n!+2n and 2log₂n+2n arithmetic exactly. 525 tests pass, zero regressions. Evals wired to ARC, HellaSwag, PIQA, Winogrande, GSM8K and seven Minerva-MATH subtasks. No training or eval numbers; that needs a GPU.

**PROBE:** Give me 200-300 words of real analysis on fully implementing this into OLMo-core and where the expected performance gains are. Grounded, accurate, not jargony.
**ANSWER:** Gains sit inside the noise floor.
**DIRECTION:** Wanted an honest verdict on whether this is worth building, not a summary.

**PROBE:** This is for the actual main branch, a plan my team wants before they decide on the implementation.
**ANSWER:** Reframed as a go/no-go RFC for the team.
**DIRECTION:** Audience is the team making a funding decision, not personal notes.
**DELIVERED:** `docs/hc-ablation/RFC-adopting-mhc.md`, ~1,050 words, commit b8939e3a. Recommends not merging yet.

---

## Standing conclusion

Don't merge mHC into OLMo-core main yet. The published gain is 0.980 vs 0.981 test loss at 7B with downstream splitting 4 of 8 benchmarks, inside seed noise. Real cost is not parameters (192-768) but 4x residual activation memory and rewriting the block tensor contract, which pulls in pipeline parallelism, expert parallelism and checkpoint compatibility, and collides with the live MoE study on `edullm/moe-m1-pilot`. The one open question worth funding: the finetuning paper found identity mixing beats learned mixing, contradicting the original mHC pretraining paper. Nobody has tested this during pretraining, and OLMo-core is a pretraining library. The harness to answer it exists and passes its correctness gate.

## Standing preferences

Verify your own output before reporting it; don't ask him to check. Lead with the objective and current state, give one recommended next action rather than a menu, keep detail hidden until asked. He consumes information at a limited rate and has lost 20-30 hour stretches to work with nothing concrete at the end, so every task needs a named deliverable.
