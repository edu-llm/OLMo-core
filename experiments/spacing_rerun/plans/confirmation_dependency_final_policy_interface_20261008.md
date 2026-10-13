# Final confirmation dependency policy interface

This interface checks eligibility for a prospective dependency-policy freeze. It does not approve source content, change primary scoring or alter the development acquisition gate. Creating a review input establishes no policy approval. No real-data final policy has been frozen by this implementation.

The supported V3 algorithm is in `configs/confirmation-dependency-algorithm-v3-r2-20261008.json`. Its algorithm digest is `91e7e9a7015fe83dae9be1c5b8ab28b6322f0971354ea75e802d2da9bac62503`. The earlier draft is retained. R2 corrects its D8 display label to the exact existing gold `optical illusion`, without an added article. It incorporates A1, explicit primary-secondary sign and descriptive interval zero-coverage divergence, and A2, duplicate-cluster and acquisition-target-form means before group means. Transfer preparation proves that only canonical duplicate roots are scored. Separate QA-root metadata completes coverage of the catalog's QA-role events. The D8 descriptive confusion table discloses that a `trick` prediction is `other` under unchanged fixed labels and normalization. This specification still needs fresh actual independent approval.

## Producer and launch guard

Functions are in `spacing_rerun.confirmation_dependency`:

```python
build_final_policy_input(
    chunks, catalog, recommendations, flagged_units, prepared_records,
    source_review_evidence, *, policy_authors, required_source_conditions=(),
    exception_protocol=None, exception_lanes=(), lane_property_reviews=(),
    protocol_coverage=None, lane_source_review_bindings=(),
)
validate_final_policy(
    policy, chunks, catalog, review_receipt, review_evidence,
    source_review_evidence,
)
freeze_final_policy(
    path, policy, chunks, catalog, review_receipt, review_evidence,
    source_review_evidence,
)
verify_frozen_policy(
    path, expected_sha256, chunks, catalog, *, manifest_sha256=None,
)
```

`chunks` is the complete final all-eight source chunk context with immutable source history and actual receipts. `catalog` is the pinned source-only catalog. Do not pass evaluation questions or the complete wrapper to policy or source authors. The launch guard may extract `source_chunks` and `source_catalog` from its existing audited wrapper.

`prepared_records` is the entire registered list of dictionaries containing `replicate_id`, `manifest_sha256`, `records` and `training_provenance`. Each record has exactly `id`, `opaque_probe_id`, `event`, `role` and `unit_id`. Every catalog duplicate root appears exactly once across transfer and QA metadata. Old and new records reference final source groups; control and QA records reference original source units. `prepared_source_metadata(manifest)` extracts those fields from the actual audited ID map. `training_record_provenance(manifest, examples, membership)` binds every packed training example and its source attribution and token counts. Each registered manifest is bound before final review. A later manifest requires fresh review and freeze.

`recommendations` supplies exact source-only dependence sets with immutable original-unit provenance. The builder projects them through final groups, builds within-event components, retains overlaps and nested sets, and includes singleton blocks. No cross-event joining or shared-answer heuristic is permitted. Original recommendations remain historical proposals. Final approval comes only from the new exact-content review.

`policy_authors` contains actual author identities accepted by `confirmation.identity`. The final reviewer must have a separate identity and task. Policy authors and source authors cannot review the final policy.

## Actual completed review evidence

Every retained source-semantic receipt, including historical negatives, has exactly one entry in `source_review_evidence`:

```text
receipt_sha256
client_request_id
receipt_file: {utf8, file_sha256, optional path}
task_status: exact actual T3 task-status result object
task_status_file: {utf8, file_sha256, optional path}
```

The status file contains `task_statuses` rows with `clientRequestId` and `result`. The entry must match exactly. Its task and latest terminal status must both be `completed`; pending child runs must be false; task, child thread and child run IDs must exist; a declared latest terminal run must match; the latest terminal summary must contain the exact receipt seal. The model must match its receipt declaration. Raw receipt bytes must parse to the unchanged receipt and match their file hash.

Original source receipts may use reviewer-local runtime labels. Preserve those labels. Their exact receipt bytes, output seal in the actual task result and actual model establish the execution binding. The source validator separately checks author-reviewer independence and any generic task-channel evidence. The binding records whether the declared label matched the actual task ID. The new final policy reviewer must name its actual client request or task ID.

`review_evidence` has the same shape for the fresh final policy receipt, without requiring a `receipt_sha256` key. No local boolean substitutes for actual completed task evidence.

## Conditional source approval rows

Each `required_source_conditions` row has:

```text
id
source_receipt_sha256
acquisition_record_ids
dependence_recommendation_ids
flagged_source_unit_ids
notes
```

Each named acquisition row must match an approved exact current-content decision in that actual source receipt. Its named dependence sets and flag mask must exist in the final membership. The final reviewer approves every exact condition row and attests that all source conditions are captured.

The checker also requires coverage of explicitly declared current approval `conditions` fields. It pins the known source04 conditional receipts `895c38af…` and `424793a4…`. The six original D7/D8 rows require their correct `chunk04/D7` or `chunk04/D8` edge and all original units in that edge in the flag mask. The latter receipt adds the two D25 synonym-label rows and requires `chunk04/D25` with its whole-unit mask. A1/A2 are enforced through the exact V3 algorithm digest. These conditions are not source approvals for other rows. A changed row requires actual review of its new content.

The adopted uniform sourceQA amendment keeps unique semantic licensing as the default. LE1a conflict, LE1b alias, and LE1c original-label nonuniqueness are conditional property classes. The methods assessment approves no rows or memberships. Each lane requires genuine source QA decisions, exact property certification, and final membership decisions below. Census findings remain candidates until those steps pass.

## Separate lane reporting specification and evidence

The reporting specification is frozen separately in `configs/confirmation-lane-reporting-policy-v2-20261008.json`. Its embedded policy seal is `bfe857888ed8a9b8fd1f771e88468344333c760fb38f0176432850bb3b98faee`; its outer prospective freeze seal is `d4dd9ffe1746ca955b022005b3bef04d7495a59adbc7c9060845256cd3ca70b0`. The real freeze time and development-informed timing are disclosed. It claims no independent approval. `confirmation_lane.reporting_policy()` checks the exact seal and definition. V3R2 remains unchanged. The earlier unreviewed reporting specification is retained, with an explicit correction that control and QA primary scores use raw original-unit aggregation rather than final source groups.

`exception_protocol` is the exact sealed adopted protocol, not the R3 methods assessment. Its frozen `data.py`, `acquisition.py` and `encoding.py` byte hashes must still match. `protocol_coverage` is required whenever that protocol is supplied. `sourceqa_protocol.verify_protocol_coverage` verifies the exact current source proof. Its `dependency_context` binds `membership_sha256`, `exception_lanes_sha256`, `required_source_conditions_sha256`, `lane_reporting_policy_sha256`, and `algorithm_sha256`, without a circular frozen-policy hash. The final input embeds this proof, and the final receipt binds its seal.

`confirmation_lane.build_exception_lane(candidate, membership, chunks)` projects no automatic acceptance. The candidate has exactly:

```text
id
event
sub_kind = conflict | alias | original_label_nonuniqueness
eligibility = LE1a | LE1b | LE1c
source_unit_ids = sorted exact narrow original-unit set
acquisition_record_ids = sorted exact narrow current QA set
dependence_recommendation_ids = sorted broader dependence-edge IDs
source_record_review_bindings = [
  {record_id, source_receipt_sha256, reviewed_content_sha256}
]
disclosure_notes
```

Every QA must have an actual current-content approved source decision. The builder binds unchanged original canonical labels, exact source-unit hashes, source group IDs, QA-to-own-unit attribution, `whole_edge_source_unit_ids`, and `dependency_block_ids`. Distinct units must have distinct label sets under the frozen normalization. Every broader edge member must be flagged, and all must share one block. The narrow set can be smaller than the dependence block, but its later final decision must explicitly resolve that difference. D25's two-row/two-unit conditional exception cannot silently replace its proposed three-row/four-unit dependence edge. Different-subject ladle units may remain in a broader block without entering the lane only if the independent final review says so.

`lane_property_reviews` contains exactly one `{lane_id, receipt, evidence}` for every proposed lane. The actual receipt has schema `spacing-confirmation-source-label-property-review-v1`, `status=approved`, exact `lane_sha256=digest(lane)`, `protocol_sha256`, matching `eligibility` and `sub_kind`, substantive `eligibility_notes`, `unresolved_items=[]`, and `ordinary_qa_defects_resolved=true`. Its `attribute_certifications` has exactly `subject`, `relation`, `modality`, `qualifiers`, `scope`, and `answer_slot`. Each has `no_faithful_nonleaking_semantic_discriminator=true` and substantive `notes`. It also requires `own_attribution_preserved=true`, `nonuniqueness_not_created_by_qa=true`, `original_labels_only=true`, `surface_correlates_assessed=true`, and a list of `surface_correlate_certifications`. Each correlate certification has a globally unique `id`, exact sorted `acquisition_record_ids` inside the lane, the corresponding exact sorted own `source_unit_ids`, `original_source_backed=true`, and substantive `notes`. Additional precise source-surface terms may be retained. The four blind-input/human flags are false, and the actual named independent `reviewer` cannot be a policy or source author. `evidence` uses the raw actual completed-task format above. No actual receipt is rewritten into this schema; a fresh genuine review must produce it.

`required_surface_correlates` derives exact K3 correlate-ID obligations from those actual bound property receipts. Final policy validation requires the coverage evaluation-custody mapping to equal them for every current QA, including explicit empty lists. The custodian must supply every required boolean; an asserted empty mapping or missing flag cannot hide a certified L5 correlate. The coverage proof independently binds exact original canonical-root counts for all catalog units.

K1 reports canonical-root counts per original unit and label, within each final group, role and replicate. For old/new roles it reports each unit's unchanged share of all scored group roots. For control/QA it reports each raw original-unit aggregation ID and its full weight inside that unit's own mean. It also reports the unchanged event-level contribution weight. For one-label pairs inside one primary aggregation unit, equal counts make the pair's contribution invariant when every pair root shares a common collapsed prediction label and that label switches between the two originals. This conditional statement does not claim invariance under arbitrary target-dependent prediction swaps, answer-NLL invariance, or semantic validity. Unequal counts disclose unequal weights; multi-label units and pairs in different primary aggregation units receive no within-group invariance claim. Nothing is reweighted.

K2 emits every unordered unit pair in every lane, with both target sides kept separate. Every existing arm, endpoint, role and saved section gets own-label, sibling-label and other counts and frequencies under existing normalization. For a pair inside a larger set, another member's label is `other` for that pair. Full frequencies use the available prediction denominator, and missing predictions make the complete table undefined while retaining available counts/frequencies and missing IDs. Absent sections remain explicitly unavailable. No pooled lane score, union gold or either-label success metric is computed.

K5 is a separately registered lane-only complement contrast with A2 means and explicit A1 sign and interval-zero-coverage divergence from the saved primary. It keeps the narrow union of accepted lane units, rather than the broader flagged set. As with other masks, an acquisition QA row is excluded if any member of its merged source group is outside that union. Empty complements remain undefined, with all registered replicate IDs retained. Existing V3R2 contrasts and ten secondary combinations remain unchanged.

## Fresh final review receipt

Final input, review, and frozen-policy schemas are version 2. The reviewer writes a new sealed artifact with schema `spacing-confirmation-dependency-final-policy-review-v2` and these fields:

```text
policy_sha256
algorithm_sha256
membership_sha256
source_bindings_sha256 = digest(policy.source_bindings)
canonical_bindings_sha256 = digest(policy.canonical_bindings)
source_review_evidence_sha256 = digest(policy.source_review_evidence_bindings)
lane_reporting_policy_sha256 = policy.lane_reporting_policy.sha256
exception_protocol_sha256 = policy.exception_protocol.sha256, or null
protocol_coverage_sha256 = policy.protocol_coverage.sha256, or null
exception_lanes_sha256 = digest(policy.exception_lanes)
lane_property_review_bindings_sha256 = digest(policy.lane_property_review_bindings)
lane_reporting_policy_approved = true
status = approved
membership_status = final_approved
unresolved_items = []
all_source_conditions_captured = true
reviewer = actual named identity
model_outcomes_seen = false
evaluation_wording_seen = false
author_rationales_seen = false
human_review_complete = false
edge_decisions
mask_decisions
condition_decisions
lane_decisions
sha256 = digest(receipt without sha256)
```

Each decision has `id`, `status`, `reviewed_content_sha256` and substantive `notes`. All decisions must be approved, and coverage must be exact with no omitted or repeated items.

For an edge decision, its ID is `edge.recommendation_id`; its reviewed content is `dict(edge, id=edge.recommendation_id)` from `policy.membership.edges`. For the two mask decisions, reviewed content is `{"id": "flagged", "source_unit_ids": policy.membership.flagged_source_unit_ids}` and the equivalent `preserved` row. Condition decisions review the exact `required_source_conditions` rows. The final membership digest binds all groups, original units, masks, duplicate memberships, canonical roots, edges and component mappings.

Each lane decision binds `digest(lane)` and repeats the exact `source_unit_ids`, `acquisition_record_ids`, `whole_edge_source_unit_ids`, and `dependency_block_ids`, with `narrow_lane_vs_broader_dependence_resolved=true`. Missing, pending, narrowed or changed sets fail. The final reviewer also must differ from every lane property review's actual task identity. The policy decision supplies no new source approval.

The reviewer should retain a negative decision and explain unresolved issues whenever any requirement fails. Those artifacts cannot pass the freezer. The schema above describes successful final review, not a template for inventing approval.

## Freeze and verification behavior

Validation rebuilds the final specification from actual source packets and checks every source and canonical binding. It calls `confirmation.validate_source_chunks` to require independently approved current source groups and QA, retaining all actual same-content negatives and valid supersession edges. Policy approval alone cannot satisfy this step.

Freezing validates first, records the real current UTC time and creates the path exclusively. An existing path causes failure. It embeds the exact review and task evidence, the source-review evidence bindings and all registered manifest hashes. It provides no way to set a backdated freeze time.

The preregistration producer binds the returned frozen `sha256` and path. At launch, `verify_frozen_policy` reads that artifact, checks its pinned seal, repeats full validation against the final actual source context and checks the launch manifest if supplied. Any changed source, membership, mask, edge, canonical record, receipt, task status or manifest fails. Source01/source04 repairs and all further source revisions must finish before final review and freeze.

## Production reporting

`saved_bundle_diagnostics(path, manifest, schedule, examples, policy, replicate_id)` reads existing acquisition and five-arm evaluator JSON after the primary completion validator passes. It checks saved rows and aggregates against exact root and source provenance. Saved transfer rows use their frozen root mapping. Both acquisition QA forms share a target cluster and average before the source group. The current evaluator's QA-teaching diagnostic scores canonical QA roots. Expanded teaching forms contribute executed exposure, but do not have separate saved prediction rows, and the reporter creates none.

`executed_presentations` checks actual log order, content hashes, loss-token counts and global steps against the frozen declaration and QA schedule. `exposure_covariates` reports every original unit and every block at each saved endpoint. Declaration records inherit all group units; acquisition QA inherits every constituent unit with its exact normalized fixed answer label; teaching records inherit the explicitly audited root unit. Generic replacements and filler have empty source attribution. Unit totals are attributed exposure. Block totals use a union of presentation IDs and count each shared presentation once. Reported clocks include optimizer steps, actual non-padding input tokens, QA gold-answer target tokens excluding EOS and filler, and loss-bearing tokens. Declaration answer-target counts are zero. Within a batch, cumulative token clocks follow frozen example order; all slots share its optimizer step. Every phase has counts, first/last clocks, all sorted between-presentation gaps, minimum/median/maximum gaps and last-presentation-to-endpoint gaps.

`d8_prediction_table` emits fixed-target by predicted-label counts for `an elaborate trick`, `optical illusion` and `other`. It creates no union-gold success score. Loss-only intermediate checkpoints have no generation predictions. Their table is explicitly undefined with missing prediction IDs; no counts are presented as complete.

`build_saved_dependency_report(bundle_inputs, frozen_policy, prereg)` revalidates the frozen policy and requires the entire registered replicate set with the existing complete five-arm confirmation receipts. Production inputs contain only `path`, `manifest_sha256`, `confirmation_audit_sha256`, `completion` and compact `primary_aggregate_binding`. `load_dependency_bundle` reloads and fully verifies one preparation, checks both exact primary-verified hashes, and releases its manifest, schedule and packed examples before loading the next replicate. Full `manifest`, `schedule` and `examples` inputs remain supported for synthetic fixtures. Exact shared audit equality is checked after verification; no guard is skipped. Only compact diagnostic rows survive across replicates. It emits every existing acquisition checkpoint, baseline, registered intermediate point and delayed endpoint. It reports canonical and paraphrase transfer separately, acquisition supervision separately and saved canonical QA teaching separately. Every existing contrast, role and EM/loss endpoint has all ten secondary mask/weight combinations, explicit A1 comparisons, retained/excluded row counts by event/group/block, and all fixed block/event influence cases. Empty or unavailable diagnostic scores remain undefined and retain their registered replicate IDs. Poorly acquired valid replicates are retained.

The separate `lane_descriptive_reporting` output binds its reporting policy and lane digest. `bundles[*].lane_root_balance` contains K1. `bundles[*].lane_prediction_tables` contains K2, alongside explicit unavailable sections. `lane_only_complement_contrasts` contains K5 and A1. No V3R2 secondary is added, removed or overwritten by this output.

Unscored catalog metadata makes the influence item set identical across role assignments. It contributes no scores or extra inference units. The inference unit remains the independently trained paired replicate. Shared acquisition outputs are repeated across arm labels explicitly and supply no extra independent data.

The existing `analyze` command uses preregistration fields `confirmation_dependency_policy_path` and `confirmation_dependency_policy_sha256`. Under the adopted protocol these canonical keys are required, alongside `sourceqa_protocol_sha256`, `sourceqa_protocol_coverage_sha256`, `lane_reporting_policy_sha256`, and `lane_reporting_specification_sha256`. `verify_preregistered_dependencies` checks all against the verified frozen policy and exact outer reporting specification. Analysis verifies them before writing the existing primary result. It then writes a separate `*.dependency-diagnostics.json` file and adds its path and digest to the primary output only after success. A secondary-report failure leaves the original primary artifact independently available and claims no completed diagnostic report. Analysis integration is outside the repository's actual A/A `NUMERICAL_SCOPE`; none of its scoped training, evaluation or A/A functions changed.

Reports record actual wall time and process peak RSS. This is a measurement, not a claim that computation is free. The first production-sized report still needs its own observed performance measurement. No numerical training code, development gate, fixed dose or primary inference rule is modified.

## Independent R1 correction contracts

The fresh review must use reporting v2, inner seal `bfe857888ed8a9b8fd1f771e88468344333c760fb38f0176432850bb3b98faee` and outer seal `d4dd9ffe1746ca955b022005b3bef04d7495a59adbc7c9060845256cd3ca70b0`. V1 remains immutable. V2 corrects K1 control/QA weights to the existing raw-original-unit primary aggregation. Old/new scoring still uses final source groups. The numerical V3R2 algorithm is unchanged. R1 was requested against v1 and therefore cannot certify this v2 amendment. Its report and disclosed earlier toy-model-test boundary deviation remain retained.

Actual final and property reviewer bindings are resolved before independence checks. The checks compare actual task, request, thread and run aliases with verified current/history source-author channels and policy-author channels, and compare final versus property channels. Fresh reviews require exact normalized concrete models. Only the `[1m]` context option is stripped. Three enumerated legacy Opus spelling aliases are allowed only for legacy source receipts with an explicit receipt-bound declaration, actual model and disclosure; the original declaration remains visible.

Legacy source review completion can use retained original child write output when its terminal summary names the full output path but abbreviates its seal. `completion_trace_file` retains the exact read-response bytes, raw SHA and `/structuredContent` pointer. The same actual completed child thread/run/model and one untruncated local command item must contain the full standalone receipt seal, raw file SHA and exact output path, plus original read-only and seal-verification output. The shared protocol helper verifies completion first. The dependency binding additionally retains the actual durable item ID/hash. A current path/hash or parent completion flag is insufficient. Receipt and status bytes are preserved. Fresh named property/final reviews still require their full seal in the original terminal summary.

`build_exception_lane_candidate` binds exact geometry and genuine retained source decision references without requiring positive source approval. It preserves every referenced status and decision in `source_decision_states`, sets `candidate_only=true` and `source_approval_verified=false`, and supplies no eligibility for freeze or launch. Candidate geometry is independently property-reviewed, then custody/coverage binds the five noncircular context hashes. Final policy receives separate `lane_source_review_bindings` for genuine positive current source decisions. Adding those approvals does not change the property-reviewed candidate digest. Final validation still requires resolved source approvals, same-content negative supersession, final policy review and full coverage. Ordinary QA defects must be resolved by actual property review; candidate construction alone cannot resolve them.

Every generic source condition must name same-event edges covering each QA's own original-label source attribution. The flag mask must cover the entire named edge, including broader-only members. A different intended scope would need an explicit independently reviewed typed contract; notes cannot infer it.

Every consumed acquisition checkpoint has a positive integer epoch, exact `acquisition-E{epoch:03}.json` filename, unique identity and global optimizer step equal to the cumulative executed `stage1_epoch` prefix. Adaptive scheduled milestones are checked when recorded in the config; fixed-dose reporting requires exactly its frozen E milestone. Raw consumed file SHA, parsed evaluation digest and exact executed prefix digest/length are retained per replicate. Saved acquisition `target_id` and integer `question_variant` must match their checked row ID; original gold comes from frozen acquisition metadata.

## Independent R2 correction contracts

Every retained source semantic receipt is resolved to its actual completed task/request/run/thread binding, then checked against verified current/history source and revision author channels and disclosed policy-author channels before final eligibility. Declared legacy `/root` labels remain visible but are not execution aliases. The existing source validator still enforces declared-name separation and demands actual task provenance for overlapping generic source author/reviewer identities. Final/property actual channel checks remain unchanged. Retained negative receipts and exact decision-scoped supersession are preserved.

The analyzer retains `primary_aggregate_binding`, a compact sealed copy of its existing five-arm primary aggregates tied to manifest SHA, original completion SHA, primary delay and exact endpoint step. This does not add or modify the original primary JSON fields or any numerical result. Production saved reporting requires that binding. At every alias of the primary endpoint, saved canonical aggregates must equal the retained original aggregates; diagnostic comparisons use that retained copy. Other endpoints still use exact original saved aggregates from verified bytes.

Saved reporting passes the original `completion` into the reader. Each consumed stage1/arm exposure log, fixed-E acquisition outcome and arm evaluation is read once as raw bytes, hashed against its canonical absolute path in `completion.files`, then parsed from exactly those same bytes. Missing, duplicate, noncanonical or stale file pins reject secondary reporting. Consumed file hashes, verified-pin booleans, completion SHA and primary aggregate-binding SHA are retained in diagnostic evidence. An internally consistent file changed after primary verification cannot silently change a diagnostic's primary comparison. The original primary artifact remains available when the secondary calculation rejects. Historical developmental acquisition files absent from a fixed-E completion receipt retain their independent raw byte/prefix evidence; production fixed-E reporting accepts only its frozen acquisition milestone and requires its completion pin.

R2's actual independent report, original failed source-channel and changed-byte reproductions, initial fixture errors, all R1 findings and prior boundary deviations remain retained. These changes have no LE1d adoption, source approval, reporting-policy version change, gate/E change, or alteration to V3R2 numerical functions or primary inference. Fresh exact-content independent review is still required.
