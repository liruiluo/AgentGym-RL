from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = _REPO_ROOT / "verl/agent_trainer/reference_policy.py"
_MAIN_PPO_PATH = _REPO_ROOT / "verl/agent_trainer/main_ppo.py"
_RAY_TRAINER_PATH = _REPO_ROOT / "verl/agent_trainer/ppo/ray_trainer.py"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "reference_policy_under_test", _POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _config(
    *,
    skip: bool = False,
    use_kl_loss: bool = False,
    kl_type: str = "fixed",
    kl_coef: float = 0.0,
):
    return SimpleNamespace(
        trainer={"skip_reference_policy_when_kl_disabled": skip},
        actor_rollout_ref={"actor": {"use_kl_loss": use_kl_loss}},
        algorithm={"kl_ctrl": {"type": kl_type, "kl_coef": kl_coef}},
    )


class SkipUnusedReferencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_policy_module()

    def test_default_preserves_reference_policy(self):
        self.assertTrue(
            self.module.should_create_reference_policy(_config(skip=False))
        )

    def test_explicit_zero_kl_mode_skips_reference_policy(self):
        self.assertFalse(
            self.module.should_create_reference_policy(_config(skip=True))
        )

    def test_rejects_skip_with_actor_kl_loss(self):
        with self.assertRaisesRegex(ValueError, "actor KL loss"):
            self.module.should_create_reference_policy(
                _config(skip=True, use_kl_loss=True)
            )

    def test_rejects_skip_with_nonzero_reward_kl(self):
        with self.assertRaisesRegex(ValueError, "kl_coef=0"):
            self.module.should_create_reference_policy(
                _config(skip=True, kl_coef=0.001)
            )

    def test_rejects_skip_with_adaptive_kl(self):
        with self.assertRaisesRegex(ValueError, "fixed KL control"):
            self.module.should_create_reference_policy(
                _config(skip=True, kl_type="adaptive")
            )

    def test_main_task_conditionally_builds_ref_role_and_logs_readback(self):
        source = _MAIN_PPO_PATH.read_text(encoding="utf-8")
        self.assertIn("should_create_reference_policy(config)", source)
        self.assertIn("if use_reference_policy:", source)
        self.assertIn("role_worker_mapping[Role.RefPolicy]", source)
        self.assertIn("mapping[Role.RefPolicy]", source)
        self.assertIn("reference_policy_enabled=", source)

    def test_trainer_guards_reference_forward_and_marks_kl_observability(self):
        source = _RAY_TRAINER_PATH.read_text(encoding="utf-8")
        self.assertIn("if self.use_reference_policy:", source)
        self.assertIn("self.ref_policy_wg.compute_ref_log_prob(batch)", source)
        self.assertIn("'trainer/reference_policy_enabled'", source)
        self.assertIn("'critic/kl_measured': float(kl_measured)", source)


if __name__ == "__main__":
    unittest.main()
