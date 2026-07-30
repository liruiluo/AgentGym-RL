from __future__ import annotations


def _get(node, key: str, default=None):
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


def should_create_reference_policy(config) -> bool:
    """Return whether PPO must construct and run the reference policy."""

    trainer = _get(config, "trainer")
    skip_when_disabled = bool(
        _get(trainer, "skip_reference_policy_when_kl_disabled", False)
    )
    if not skip_when_disabled:
        return True

    actor_rollout_ref = _get(config, "actor_rollout_ref")
    actor = _get(actor_rollout_ref, "actor")
    if bool(_get(actor, "use_kl_loss", False)):
        raise ValueError(
            "Cannot skip the reference policy while actor KL loss is enabled"
        )

    algorithm = _get(config, "algorithm")
    kl_ctrl = _get(algorithm, "kl_ctrl")
    kl_type = _get(kl_ctrl, "type")
    kl_coef = _get(kl_ctrl, "kl_coef")
    if kl_type != "fixed" or kl_coef is None or float(kl_coef) != 0.0:
        raise ValueError(
            "Skipping the reference policy requires fixed KL control with "
            "kl_coef=0"
        )
    return False
