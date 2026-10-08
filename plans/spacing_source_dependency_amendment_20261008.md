# Scale-source dependency amendment, October 8, 2026

The source-only Opus 5.5 preservation adjudication identified two sets of overlapping propositions in development scale events event_089 and event_097. Four source units have `preserve_source_ambiguity` status. The existing protocol requires each flagged unit to remain a literal raw singleton, with its original canonical gold and probes. The flag preserves an unresolved source limitation. It does not establish independence between propositions.

Keep the original separate groups as the primary estimand. Before any scale acquisition grid starts, freeze the two dependency sets as analysis metadata outside the authored rows and require a separate equal-cluster sensitivity report for every scale acquisition dose. This implements the primary per-group choice and the contrasting secondary choice requested by the actual R5 review. It does not claim the reviewer has approved this amendment. Independent review and human review remain incomplete in the policy record.

The policy's actual freeze is `2026-10-08T08:13:02.834250+00:00`. Its canonical SHA-256 is `948afc3ee5d9e08aef5c60515e80c6ec415f0a1c890e12964058abde2c7ddd5d`. Six normal development acquisition grids in job 1806319 had completed, and their E2, E3, E4, E6 and E8 results had been observed before this amendment. No scale acquisition grid had started. The prospective scope is the three scale grids. This record makes no claim of blinding to the normal-grid outcomes.

The machine-readable policy is [source-dependency-policy-20261008.json](../experiments/spacing_rerun/configs/source-dependency-policy-20261008.json). It binds source catalog `aff2b524e93eb0b19d3b0f1ee00d811904e35d5a2ee89c730a63dcb154818d56`, final authored source02v7 `9596bcffaefe08f85ee873c4de8898eb993b94121278a5599e06d8ebfe1a59d5`, frozen acquisition core `56d0d72fcfbff828dafe27e6713041fc9b0d6ed0ad9de2a799c312083d9975cc`, and actual R5 receipt `b2d9dc90d58614bc13acf91d4219feaf9f8a4697114fccda91fb9628bd25551d`. Each member binds its full authored group hash, declaration projection, exact raw source assertion, source-unit hash, preserved status and canonical answer labels.

| Set | Event | Grounded group IDs |
| --- | --- | --- |
| SET8 | event_089 | ground-14b2639e9d56503f1c89ef45, ground-5905b61da0f98fdcc8a551b1, ground-06203eaf61fae56128649c15 |
| SET9 | event_097 | ground-bb16f9464eed5e09a0adca54, ground-a5c177ecc48b796682b1a689, ground-332f3165bf1a3cf84902c04b |

All six groups must remain old within their original events in every scale manifest. The two sets cannot overlap or cross an event or role. All undeclared old groups each form one singleton dependency cluster.

For both exact match and gold answer loss, the primary score continues to average probes within a group, groups within an event, then events equally. The separate sensitivity averages probes within a group, group means within each declared dependency cluster, clusters equally within an event, then events equally. It retains every old probe and uses the original canonical answer acceptance and answer-token NLL excluding EOS. Group members receive equal weight within their cluster, regardless of how many probes each group contains. This is a change in within-event estimand weighting. Independent experimental replicates remain the inferential units.

The sensitivity cannot alter the 40–70% gate, dose eligibility, the frozen common-E rule or arm eligibility. The six normal and three scale grids still require the original primary score to pass at a common dose; eligible doses still use the scaled primary mean closest to 55%, with ties going to the lower dose. Report primary and secondary EM and loss for all E2, E3, E4, E6 and E8 scale evaluations, including their secondary-minus-primary differences and per-event secondary scores. Report the declared overlap even if the weighting difference is zero. No source statement, canonical gold, alias, probe, schedule, exposure count, model or training/evaluation numerical function changes.

The compiled schedule assigns starts per group. The mandatory pre-grid binding therefore records every member's actual prepared start and labels each set `shared` or `mixed_schedule`. A scale probe performs acquisition only and never executes that compiled stage2 schedule or enters spacing-effect analysis. Any future use of these sets in a review-arm experiment must retain the mixed-schedule disclosure and prospectively define its interpretation before that experiment's outcomes. This amendment authorizes the acquisition sensitivity only.

The external diagnostic lives in [dependency_sensitivity.py](../experiments/spacing_rerun/spacing_rerun/dependency_sensitivity.py). It runs alongside the unchanged numerical snapshot. After fixing the final remote policy, authored source and prepared paths, create one pre-grid receipt for each scale manifest before submitting its GPU job:

```bash
PYTHONPATH=experiments/spacing_rerun python -m spacing_rerun.dependency_sensitivity bind \
  --policy POLICY.json --prepared PREPARED \
  --final-source chunk-02-v7.json --output dependency-pregrid-binding.json
```

The command records the actual current UTC, exact prepared manifest and schedule hashes, source and policy file hashes, and actual group placements. Retain this receipt with the submitted script and launch evidence. The receipt is an external protocol binding because adding a config key would change the frozen acquisition core's exact config frame. The policy and receipt must be present at their final paths before training. The operator must not create a receipt later and backdate its fields.

After the unchanged scale calibration finishes, use the separately committed diagnostic code and all nine actual bound prepared manifests:

```bash
PYTHONPATH=experiments/spacing_rerun python -m spacing_rerun.dependency_sensitivity report \
  --policy POLICY.json --prepared PREPARED \
  --pregrid-binding dependency-pregrid-binding.json \
  --manifest-binding nine-manifest-binding.json --stage1 STAGE1 \
  --output dependency-grid-sensitivity.json
```

The report requires both the policy freeze and its actual pre-grid receipt to precede the grid's recorded start. It verifies the exact actual nine-manifest binding, core and trial identity, complete unselected E2/3/4/6/8 decision and evaluations, raw source and role membership, all canonical probe coverage, original exact-match acceptance and answer-token loss. It streams each of the five dose checkpoint files once for its SHA-256 and inspects full-state metadata through sequential CPU memory mapping. It checks each evaluation against the checkpoint epoch, completed epoch cursor, global step, acquisition QA tokens and target/variant exposure counters. It recomputes the original primary EM/loss and checks the stored aggregates and decision gate before writing the separate secondary report. Both commands reject existing output paths. Run remote verification under Slurm. Every report retains the actual input file hashes, total checkpoint bytes hashed, elapsed verification time and peak process RSS. Linux reports also retain the measured process I/O counter deltas, including bytes read from storage and bytes returned by read calls. Read one checkpoint at a time and release its tensor mapping before the next checkpoint. An invalid or missing sensitivity report leaves this amendment's required diagnostic work incomplete; it cannot authorize a different dose or silently remove a failed scale trajectory.
