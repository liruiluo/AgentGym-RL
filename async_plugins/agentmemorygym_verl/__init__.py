"""Thin AgentMemoryGym extensions for upstream veRL.

The package itself is dependency-light so launch/finalization checks can run
without importing torch.  veRL imports :mod:`agentmemorygym_verl.action_gae`
explicitly through ``VERL_USE_EXTERNAL_MODULES`` to register the estimator.
"""

__all__: list[str] = []
