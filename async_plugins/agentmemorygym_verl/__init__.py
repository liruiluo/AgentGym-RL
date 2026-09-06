"""Thin AgentMemoryGym extensions for upstream veRL.

The package itself is dependency-light so launch/finalization checks can run
without importing torch.  veRL imports :mod:`agentmemorygym_verl.token_gae`
explicitly through ``VERL_USE_EXTERNAL_MODULES`` to register the active SAO +
CompactionRL estimator.  The older action-axis estimator remains available
only as the matched r112 action-level comparator implementation.
"""

__all__: list[str] = []
