# P4: Validating learning science for LM training (paper workspace)

**Start here:** [progress.md](progress.md) has the current status, the open checklist and a dated history. If you are picking this project up, read [HANDOFF.md](HANDOFF.md) first; it explains what to ignore.

| Folder | What's in it |
|---|---|
| [original/](original/) | The team's files as received, **untouched**. Contents: the two `.docx` reports, the one-pager (`onepager/`), the phase-2 architecture ideas (`other/`), and the synthetic-student documents (`out_of_scope_synthetic_student/`, not part of this work). |
| [paper/source/](paper/source/) | Markdown copies of the two reports, used as the paper sources. [P4_whitepaper_writeup.md](paper/source/P4_whitepaper_writeup.md) covers spacing; [P4_final_report.md](paper/source/P4_final_report.md) covers interleaving and mastery gating (the synthetic-student section is omitted). Not edited yet. |
| [paper/drafts/](paper/drafts/) | New writing. [P4_spacing_paper_sections.md](paper/drafts/P4_spacing_paper_sections.md) is a collage of [PILOT] / [BOTH] / [RERUN] versions of each section, to assemble into a paper later. |
| [paper/figures/](paper/figures/) | Figures extracted from the reports. |
| [plans/](plans/) | Experiment PRDs: [spacing and interleaving reruns](plans/PRD_spacing_interleaving_reruns.md) and [mastery-gated curricula](plans/PRD_mastery_gated_curriculum.md). |
| [research/](research/) | [research.md](research/research.md): literature synthesis for all three studies, including the 10 papers to add per study and APA references with URLs. [methods_verification.md](research/methods_verification.md): the papers checked against the team's GitHub code. [notes/](research/notes/): detailed notes 00–11, each starting with a TL;DR. Note 11 surveys the org's compute pipelines. |
| `.claude/skills/scientific-writing/` | The project writing skill (local only; listed in `.gitignore`, so it isn't in the repository). It loads automatically when paper prose is written or edited, and its checker is `check_prose.py`. |

**Conventions**
- Every Markdown file we write starts with a TL;DR.
- Paper sources are edited only after the team gives the go-ahead.
- The team's GitHub code is read remotely and never stored in this folder.
