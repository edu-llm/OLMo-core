# Handoff for the next agent or teammate

**Read first:** [progress.md](progress.md) (status, open checklist, dated history), then [README.md](README.md) (folder map).

## Ignore all synthetic-student material
The team is **not** doing the synthetic-student work. Don't use it, cite it, summarise it or carry it into any paper, plan or note. It still appears in two unmodified originals, which are kept on purpose:
- `original/P4 Validating Learning Science - Final Report.docx`: Test 04 is the synthetic-student section. Skip it. Our markdown copy, [paper/source/P4_final_report.md](paper/source/P4_final_report.md), already omits it.
- `original/onepager/P4_onepager.html`: one synthetic-student paragraph. Skip it.

The separate synthetic-student documents are not in this repository.

## Where things are
| Need | Go to |
|---|---|
| The papers being turned into publications | [paper/source/](paper/source/): spacing whitepaper; Final Report (Test 02 interleaving, Test 03 mastery gating) |
| New paper text to assemble from | [paper/drafts/P4_spacing_paper_sections.md](paper/drafts/P4_spacing_paper_sections.md) ([PILOT] / [BOTH] / [RERUN] section versions) |
| Experiment plans | [plans/](plans/): spacing and interleaving reruns (Part 3 covers implementation independent of the compute pipeline); mastery-gated curricula |
| Literature, verified citations (APA with URLs), top-10 papers to add | [research/research.md](research/research.md) |
| What the code actually did vs. what the papers say | [research/methods_verification.md](research/methods_verification.md) |
| Detailed notes 00–11 (incl. 11: compute pipelines) | [research/notes/](research/notes/) |
| Team originals, never edited | [original/](original/) |

## Working rules
- **Don't edit `paper/source/` without the team's go-ahead.** Write new text in `paper/drafts/`.
- **Writing:**
  - no em-dashes;
  - no hype or filler phrases;
  - claims no stronger than the evidence;
  - every number traceable to code, logs or a verified citation;
  - `[CITE:]`, `[TODO:]` or `[CHECK:]` placeholders instead of guesses.

  (These rules come from a local writing skill that isn't committed here.)
- **Never reproduce the known errors** listed in progress.md's checklist and in research/methods_verification.md (e.g. misused citations; the pilot evaluation set described as "held-out paraphrases").
- **Every Markdown file starts with a TL;DR.** Keep progress.md's checklist and history up to date, and keep links relative.
- **Experiment code** lives on other branches of this repository (`anshulm/fictionalqa-review`, `p4/blocked-vs-interleaved`). The AWS platform is retired; see research/notes/11.
