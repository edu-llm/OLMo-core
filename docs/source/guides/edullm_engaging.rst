Engaging with eduLLM
====================

Pilot status
------------

The eduLLM pilot is operational. The core request-to-results flow passed
end-to-end on Issue #8, so teammates may submit reviewed requests; earlier
draft guidance that said to wait for Issues or pilot activation is obsolete.

Teammate and approver onboarding
--------------------------------

Teammates clone or update the repository, then open it in Cursor:

.. code-block:: bash

   git clone https://github.com/edu-llm/OLMo-core.git
   cd OLMo-core
   git switch main
   git pull --ff-only origin main
   cursor .

For an existing clone, start with ``git switch main``. Approvers use normal
GitHub pull-request reviews and do not run operator setup.

Operator onboarding
-------------------

Operators clone or update the current ``main`` branch, install the W&B-enabled
CLI, authenticate GitHub and W&B, and then run:

.. code-block:: bash

   git clone https://github.com/edu-llm/OLMo-core.git
   cd OLMo-core
   git switch main
   git pull --ff-only origin main
   python -m pip install -e '.[wandb]'
   gh auth login
   wandb login
   # Connect to the MIT VPN here if your network requires it.
   edullm setup --orcd-username YOUR_MIT_USERNAME
   edullm jobs --mine

Replace ``YOUR_MIT_USERNAME`` with the operator's MIT username. Operator Slack
member IDs are centrally configured in the reviewed roster; ``edullm setup``
never asks for a Slack ID.

Meric (``meric233``) and Amy Lin (``alsy7009``) are staged in the roster but
are not assignment-enabled. Each must report successful personal setup before
assignment can be enabled. Enabling either operator requires a later, tiny,
reviewed configuration change.

Prepare and review
------------------

The submitting teammate owns the experiment branch and pull request, including
the training code, configuration, data choices, and success metrics. A team
lead approves the exact current pull-request head SHA only after the required
CI checks pass. A direct-main SHA is not eligible.

Submit with /submit-edullm-job
------------------------------

Run the real ``/submit-edullm-job`` Skill after the pull request is ready. The
Skill validates the request, shows the canonical Issue preview, asks for
confirmation, and creates the structured Issue without changing its validated
body. Manually completing the Issue form does not satisfy acceptance.

Assignment
----------

GitHub Actions validate the Issue without compute credentials, assign one
enabled operator, and send that operator one Slack assignment notification.
The workflow receives only GitHub's scoped token and the required assignment
webhook; ORCD, SSH, Kerberos, Duo, W&B, and S3 credentials stay operator-side.

Operate
-------

Operators use:

* ``edullm setup`` to configure and verify the local operator environment.
* ``edullm run`` to accept the oldest assigned eligible request, revalidate it,
  and submit it without a separate manual audit prompt.
* ``edullm jobs [--mine]`` to list authorized requests and reconcile scheduler
  state.
* ``edullm logs ISSUE`` to read the authorized bounded redacted log.
* ``edullm stop ISSUE`` to stop an authorized recorded job idempotently.
* ``edullm logout`` to close only the project SSH ControlMaster.

Identity and safety
-------------------

There are no shared credentials. Requests cannot use a direct-main SHA or
supply shell text. The approved exact SHA is checked during validation and
again immediately before submission. Structured arguments are shell-quoted,
and the idempotent submission transaction permits exactly one ``sbatch``.

Status and tracking
-------------------

Each attempt has a deterministic W&B run ID and URL derived from its Issue,
attempt, and Slurm job ID. Training emits the real experiment metrics to that
run. ``edullm jobs`` reads ``squeue`` and ``sacct`` evidence and maps terminal
Slurm state back to the GitHub Issue.

Deferred
--------

The initial slice does not include scheduled W&B monitoring from the original
Plan 2 Task 8, Plan 3 work, S3, Apptainer, assignment enablement for staged
operators, advanced Slack reminders or terminal threads, strict ruleset
automation, or broad rollout polish.
