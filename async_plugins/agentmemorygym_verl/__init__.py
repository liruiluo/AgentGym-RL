"""Thin AgentMemoryGym extensions for upstream veRL.

The package itself is dependency-light so launch/finalization checks can run
without importing torch.  veRL imports :mod:`agentmemorygym_verl.token_gae`
explicitly through ``VERL_USE_EXTERNAL_MODULES`` to register the fresh Hybrid
+ token-GAE estimator.  The older action-axis estimator remains available as
the unchanged Hybrid baseline implementation.
"""

__all__: list[str] = []
