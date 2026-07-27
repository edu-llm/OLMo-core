# Research brief: can four 400M peers beat distillation from a larger teacher?

Research cutoff: 2026-07-25

## Bottom line

Yes, this is worth testing—but as a high-risk claim with unusually strong controls, not as a result the literature already predicts.

The strongest defensible sell is:

> **Four complementary OLMo-compatible 400M peers may produce a better independently deployable 400M model than four otherwise identical 400M students distilled from a demonstrably stronger approximately 1B teacher, especially on verifier-valid strategy coverage and sealed compositional generalization.**

The words in that sentence matter. The comparison is not four peers versus one teacher student; it is four students versus four students with symmetric selection. The teacher is not merely larger; it must prove stronger on two calibration shards before the championship. The endpoint is one selected 400M model, not an oracle, router, or debate ensemble. “Novel” means objectively valid and absent from matched fixed-budget pre-peer and teacher output banks. Beating the raw larger teacher is a moonshot, separate from the core claim that peer learning beats larger-teacher *distillation into the same 400M deployment class*.

No located primary establishes this complete proposition. That is a reason the experiment could matter, not a reason to assume it will work.

## Agreement: what the evidence does support

### A distilled student can cross a teacher's aggregate score

A teacher is not a mathematical performance ceiling. [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) reports a 770M T5 student exceeding few-shot PaLM 540B on a benchmark while using task labels and teacher rationales. [On-policy GKD](https://arxiv.org/abs/2306.13649) reports a same-architecture self-distilled student above its teacher on GSM8K. These are real teacher-crossing precedents.

They are not evidence of peer-created knowledge. Task-specific training, gold labels, rationale supervision, regularization, or a better adaptation policy can raise a student's average while it merely trades errors with the teacher. Neither paper reports the item-level conjunction required here: post student correct, all pre peers wrong, and larger teacher wrong under equal attempt budgets.

### Complementary peer transfer is plausible

The closest direct LLM result is [OPCoD](https://arxiv.org/abs/2606.14368), which starts from paired Qwen3-8B science specialists and reports mutual improvements with gated, on-policy peer feedback. Its tutors are admitted only when they have demonstrated relevant knowledge, and its measured break rate is lower for admitted than excluded tutors. That supports three choices in the notebook: establish complementarity first, freeze models within an exchange round, and route only verifier-confirmed rescue signals.

Two independent peer-reviewed vision studies support a narrower topology claim. [Deep Mutual Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) and [OKDDip](https://doi.org/10.1609/aaai.v34i04.5746) show peer-trained compact classifiers beating compact students from conventional larger-teacher KD in reported settings. In both, the larger teacher itself remains stronger. This is exactly why the core sell should be “better 400M training topology” before it becomes “better than the raw teacher.”

### Multiple roles can preserve useful diversity longer than single-agent self-training

[Multiagent Finetuning](https://arxiv.org/html/2501.05707v2) clones a base model into generation and critic roles, trains them on separate accepted response sets, and reports continued gains over repeated rounds after its single-agent loop plateaus or falls. The paper's MATH results report Phi-3 rising from 58.8% to 66.0% and Mistral from 22.5% to 28.2% over five iterations. It also reports higher response-diversity measures than its single-agent comparator.

That is meaningful support for the proposed mechanism. It also exposes the main deployment caveat: its evaluated object remains a multi-model, multi-round debate and majority-vote system. [ReConcile](https://aclanthology.org/2024.acl-long.381/) likewise reports that heterogeneous multi-model deliberation can exceed GPT-4 on three of seven studied datasets, but it spends repeated inference and uses multiple model families. These studies support a population result; they do not show that the gain compresses into one small member.

### The fairest primary teacher is probably near 1B, not automatically 7B

The decisive teacher-size evidence comes from the peer-reviewed [capacity-gap law for language-model distillation](https://aclanthology.org/2025.acl-long.1097/). Its fitted relation is approximately `teacher size = 2.498 × student size − 11.498M`; substituting 400M gives about 987.7M. Independent autoregressive-LM experiments also report non-monotonic teacher-size effects: [Revisiting Knowledge Distillation](https://arxiv.org/abs/2402.11890) finds that its largest OPT teacher is not the best teacher for OPT-125M under all objectives. [Strong Teacher Not Needed?](https://arxiv.org/abs/2605.23857) reports that at its 300B-teacher-token condition downstream transfer declines from +4.3% with a 1.7B teacher to +3.5% with 3.8B and +2.8% with 8B, although other configurations generally improve with teacher size.

This does not prove that an OLMo 2 1B checkpoint is optimal for the team's custom 400M model. It gives the best available prior. Use a checkpoint-stage-matched approximately 1B OLMo-compatible model as the primary larger teacher. Treat 7B—about a 17.5× nominal gap—as a separately gated stress test. A 7B-only comparison risks testing a capacity-mismatch failure of KD rather than the value of privileged external knowledge.

Official revision-pinned OLMo 2 1B and 7B tokenizer artifacts were directly checked and were byte-identical at the inspected revisions, including the 7,137,177-byte `tokenizer.json` SHA-256 `73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca`. This says nothing yet about the local 400M tokenizer. Ordinary tokenwise KL is enabled only after the local token-to-ID maps, vocabulary, special IDs, tokenizer pipeline, and probe encodings match exactly. Cross-tokenizer sequence/rationale distillation is a different treatment.

### Creativity must be validity-first

[CreativityPrism](https://openreview.net/forum?id=3pfsQcEtNC) and [NoveltyBench](https://arxiv.org/abs/2504.05228) explicitly separate quality, novelty, and diversity; their results show those dimensions do not necessarily move together. [Strategy Diversity](https://arxiv.org/abs/2605.09292) reports that models with very high answer accuracy still recover fewer mathematical strategies than a human reference inventory. [Ordered CommonGen](https://aclanthology.org/2025.acl-long.1508/) shows that changing required concept order can expose structural generalization failures even when ingredients are held fixed.

So the primary creativity-adjacent endpoints should be objective: executable code, exact mathematics, constrained constructions, or other independently verifiable tasks with multiple valid strategies. Count a novel strategy only after validity. Use a frozen structural taxonomy or blinded adjudication. Embedding distance, verbosity, and unusual phrasing are descriptive; they are not creative success.

## Disagreement and live uncertainties

### Does a stronger teacher help more?

Large-teacher successes such as [Speculative Knowledge Distillation](https://arxiv.org/abs/2410.11325) and Distilling Step-by-Step show that powerful teachers can deliver valuable supervision. The capacity-gap literature says transfer can reverse when the teacher is too far from the student. Both can be true: a stronger model may contain better information while producing a distribution that a 400M student cannot efficiently absorb. The experiment should settle this locally by including the approximately 1B primary and, only after an absorption pilot, a 7B secondary arm.

### Does diversity cause quality, or merely buy more lottery tickets?

Peer and multiagent studies report diversity alongside gains, but repeated sampling alone can change apparent capability dramatically. [HumanEval](https://arxiv.org/abs/2107.03374) shows a large gap between pass@1 and pass@100, and [Large Language Monkeys](https://arxiv.org/abs/2407.21787) shows coverage continuing to rise over very large sample budgets while practical selection can lag. A peer population receiving four models times `k` samples cannot be compared with one greedy teacher output. `large_teacher_diverse` must receive the full attempted-output budget and the same accepted verified-target count as the peer arm.

### Can four models outperform because four were trained and selected?

Yes. Best-of-four selection is an optimization budget. [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html) show that selection overfitting can be comparable to differences between algorithms. The remedy is structural: train four students in both policies, use the same tuning and checkpoint opportunities, select one from each arm on an independent development split, evaluate once on the untouched final set, and publish all four outcomes.

### Does a cohort win imply one model learned the cohort's knowledge?

No. [Branch-Train-MiX](https://arxiv.org/html/2403.07816v1) combines specialist feed-forward blocks into a routed mixture-of-experts and reports aggregate performance above a Llama-2-13B comparator in its table, but the resulting model retains expanded expert capacity. ReConcile and Multiagent Finetuning retain multiple inference agents. These are valid systems results. They cannot support the single-400M headline unless one selected member wins independently.

## Best-shot experiment

### Stage 0: prove the ingredients exist

Create four specialist warm starts from the team's OLMo-compatible 400M checkpoint, then reuse that exact quartet checkpoint-by-checkpoint in every arm. The quartet must pass preregistered gates: task performance away from floor and ceiling, best-of-four oracle headroom, and unique-correct mass from every member. Different random seeds without complementary correct answers are not the hypothesized mechanism.

Pin the approximately 1B teacher and its checkpoint stage. On an election shard, identify the best warmed peer. On a separate confirmation shard, require the frozen teacher to exceed that peer by a practical point margin with a positive paired lower bound. Do not use the final test to find a teacher that looks strong.

### Stage 1: isolate peer information

Run a small excluded-seed mechanism screen:

- `peer_frr_onpolicy`: a frozen, verifier-confirmed peer scores only rescue-eligible states visited by the target student's own rollout;
- `self_snapshot_op`: the student's frozen snapshot scores the same eligible on-policy states;
- `gold_private_equal_cost`: ordinary verified training spends the peer system's generation budget productively.

Continue only if cross-peer information improves over both controls without unacceptable specialty forgetting, shift loss, general-language NLL degradation, or complementarity collapse. This stage does not prove the larger-teacher sell; it prevents spending the full budget when the peer mechanism itself is absent.

### Stage 2: four versus four

For every paired population-training seed, fork the same four warmed checkpoints into:

1. `peer_frr_onpolicy`;
2. `large_teacher_single`: the fixed approximately 1B teacher supplies on-policy token KL on the same student states when tokenizers and checkpoint stages match; otherwise the arm is explicitly a single-sequence fallback;
3. `large_teacher_diverse`: the teacher receives eight attempts per item, matching two rollout banks across four peers, and supplies verifier-passing sequence/rationale targets;
4. `gold_private_equal_cost`;
5. `self_snapshot_op`.

All arms receive identical hard examples, private replay, student seeds, optimizer/update schedules, tuning-trial budgets, checkpoint cadence, selection policy, and final audit. Match accepted auxiliary-target counts by student and round. Publish auxiliary target-token differences rather than claiming answer-only KL and rationale CE expose identical information.

One fairness view cannot answer both causal and economic questions. Report:

- an exposure-matched topology comparison, holding hard data, accepted targets, student schedule, and selection fixed; and
- a quality-versus-all-in-compute frontier that counts every teacher/peer forward, student forward/backward, decoded and rejected token, verifier call, failed/tuning run, selection evaluation, and inference attempt.

### Stage 3: sealed novelty audit

The full untouched objective-validity score is primary. Then build fixed-`k` output banks with identical temperature, top-p, seeds, prompt history, token cap, tools, stop rules, verifier, and rejection policy.

For stringent teacher-external novelty, each pre peer receives `k` samples and the raw teacher receives `4k`, matching the total population lottery. A post output counts only if valid and neither bank contains a valid solution. Report valid strategy coverage at fixed `k`, strategy families absent from every pre/teacher bank, and withheld specialty compositions plus structural/order counterfactuals. Do not define the novelty subgroup from one greedy teacher miss.

Open-ended ideation and writing remain secondary. Blind model identity, randomize and swap pair order, length-match, use an evaluator family absent from training, validate it on preregistered human ratings, and report usefulness/quality, originality, and diversity separately. [LLM-as-a-judge research](https://arxiv.org/abs/2306.05685) documents position and verbosity bias; [self-preference work](https://arxiv.org/abs/2404.13076) documents model-family affinity.

## Statistical and deployment gates

One four-student population-training seed is one replicate. The four students and thousands of test items do not multiply the outer sample size. Estimate paired-seed variance on excluded pilots, freeze a practically meaningful superiority margin and replication count, and use paired seed-level inference. The core success gate is a one-sided lower confidence bound above that margin for selected `peer_frr_onpolicy` versus selected `large_teacher_diverse`, while also beating `large_teacher_single`, ordinary training, and self-snapshot and satisfying retention/shift bounds.

Use this success hierarchy:

1. peers beat ordinary training and self-snapshot;
2. peers beat the single/on-policy larger-teacher student policy;
3. peers beat the diverse larger-teacher student policy;
4. peers produce more valid outputs missed by all pre peers and the larger teacher;
5. moonshot: the selected peer-trained 400M beats the raw larger teacher under matched inference.

Report the selected single model, four-model oracle/router, and full collective separately. A router result may be useful, but it spends population capacity. A selected-model result is the sell.

## Unverified or single-origin propositions

- OPCoD is the closest direct peer LLM evidence but is a recent preprint, pairwise rather than four-agent, and uses 8B specialists. Its transfer to 400M is unknown.
- The `~2.5×` capacity-gap fit is peer-reviewed but induced mainly from GPT-2/Pythia configurations, not this OLMo-compatible lineage. Approximately 1B is a prior to validate, not an oracle.
- Strategy Diversity, NoveltyBench, and several creativity studies are recent and not yet independently replicated at scale.
- The idea that peer interaction creates genuinely teacher-external strategies—rather than consolidating complementary pre-existing solutions—has no direct primary demonstration in the located literature.

## Gaps and disconfirmation

No located study compares four OLMo-compatible 400M peers with four matched approximately-1B-teacher-distilled 400M students under symmetric selection and complete cost accounting. No located study publishes the full pre-peer/teacher/post-student correctness tensor needed for matched teacher-external novelty.

The hypothesis is disconfirmed, not rescued rhetorically, if any of the following happens:

- the peer advantage disappears against `large_teacher_diverse`;
- it exists only for the best of four and not the four-run distribution;
- it exists only on a teacher-greedy-failure subset, not the full test or matched fixed-`k` bank;
- diversity increases while valid utility, shift, or retention falls;
- only the oracle/router/ensemble wins, while no single 400M does;
- the approximately 1B teacher fails the preregistered strength gate;
- the peer arm loses on the all-in-compute frontier after cheaper alternatives spend their budgets productively.

Finally, research channel coverage was slightly narrowed. Fourteen of fifteen configured channels were live, but Brave was degraded because no API key was configured. GitHub and arXiv also intermittently rate-limited some branch retrievals. The investigation routed through other indexes and direct primary reads, but exhaustive independent-web coverage cannot be claimed.
