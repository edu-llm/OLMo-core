# P4 confirmation catalog: final-pass amendment (2026-10-13)

Status: chosen by the study owner on 2026-10-13, after the round-3 outcomes were known. No model output or
confirmation outcome has been observed. Reason: the confirmation code requires an approved record for every
source-QA target and every evaluation probe, so UNRESOLVED records cannot be retained or dropped.

Scope: the 29 records UNRESOLVED after round 3: 22 source-QA records (Channel A) and 7 evaluation paraphrases
(Channel B). No approved record is reopened.

Rule:
1. A stronger author, Claude Opus, rewrites each in-scope record once. It is a separate agent per channel, and
   the channel separation still holds: the Channel A author never sees evaluation questions.
2. New allowance: where two answer labels in one event cannot otherwise be told apart, a source-QA question may
   name the competing label as an explicit contrast (for example "...other than environmental activism?").
   The question still must not contain its own answer or a near-synonym of it.
3. The FarmShare mechanical checks run.
4. A fresh Claude Sonnet reviewer checks each record against the same criteria. This reviewer took no part in
   earlier rounds for those records.
5. Any record that still fails goes to the study owner for a decision. This pass does not mark it UNRESOLVED.

Disclosure: added after seeing which records failed rounds 1–3. It changes the author model and adds the
contrast allowance, and it applies only to those records.
