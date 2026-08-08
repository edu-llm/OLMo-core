"""
Hybrid hyperparameter-optimization (HPO) controller for OLMo-core.

This package implements a deterministic, resumable HPO controller that combines an
official FT-PFN freeze-thaw surrogate with MFPI-random allocation (ifBO), a BTTackler
diagnostic cutoff layer, clean-room IPBT population mechanics, and a Centaur-pattern
LLM overlay.

.. important::
    The submodules here are intentionally import-light. Pure controller logic
    (:mod:`~olmo_core.hpo.types`, :mod:`~olmo_core.hpo.objective`, :mod:`~olmo_core.hpo.state`,
    :mod:`~olmo_core.hpo.ifbo`, :mod:`~olmo_core.hpo.bttackler`, :mod:`~olmo_core.hpo.ipbt`,
    :mod:`~olmo_core.hpo.centaur`) depends only on ``numpy`` and the standard library so it can
    be unit-tested without ``torch`` or the optional ``hpo`` third-party dependencies
    (``ifbo``, ``cmaes``, ``openai``). Modules that drive a real training run
    (:mod:`~olmo_core.hpo.worker`) import ``torch``/``olmo_core.train`` lazily.
"""
