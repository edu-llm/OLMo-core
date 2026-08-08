# edullm-datasets agent skill

`skill/SKILL.md` is the **canonical, package-shipped** copy of the skill — edit it here.

It is mirrored to **`.claude/skills/edullm-datasets/SKILL.md`** at this repo's root so that a Claude
Code session launched with `edullm-data` as its working directory discovers the skill automatically
(skill discovery walks from cwd up to the *repository* root — and this repo is its own git root, so a
copy in a parent directory is out of scope). Keep the two in sync; `skill/SKILL.md` is authoritative:

    cp skill/SKILL.md .claude/skills/edullm-datasets/SKILL.md

To install the skill into another repo, copy it there too:

    cp skill/SKILL.md <other-project>/.claude/skills/edullm-datasets/SKILL.md
