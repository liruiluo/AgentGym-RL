from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_PATH = _REPO_ROOT / "scripts/agentmemory/verify_fsdp_checkpoint.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "agentmemory_verify_fsdp_checkpoint_under_test", _TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_checkpoint(root: Path, *, step: int = 10, world_size: int = 2) -> None:
    (root / "latest_checkpointed_iteration.txt").write_text(
        f"{step}\n", encoding="utf-8"
    )
    step_directory = root / f"global_step_{step}"
    step_directory.mkdir(parents=True)
    (step_directory / "data.pt").write_bytes(b"data")
    for role in ("actor", "critic"):
        role_directory = step_directory / role
        role_directory.mkdir()
        for stem in ("model", "optim", "extra_state"):
            for rank in range(world_size):
                (role_directory / f"{stem}_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                    f"{role}-{stem}-{rank}".encode("ascii")
                )


class VerifyFsdpCheckpointTests(unittest.TestCase):
    def test_complete_actor_critic_checkpoint_passes(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root)
            result = tool.verify_checkpoint(root, step=10, world_size=2)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["tracker"], 10)
            self.assertEqual(result["roles"]["actor"]["model"]["ranks"], [0, 1])
            self.assertEqual(result["roles"]["critic"]["extra_state"]["files"], 2)

    def test_numeric_rank_order_supports_double_digit_world_size(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root, world_size=12)
            result = tool.verify_checkpoint(root, step=10, world_size=12)
            self.assertEqual(
                result["roles"]["actor"]["model"]["ranks"], list(range(12))
            )

    def test_tracker_mismatch_fails(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root)
            (root / "latest_checkpointed_iteration.txt").write_text(
                "9\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                tool.CheckpointVerificationError, "tracker mismatch"
            ):
                tool.verify_checkpoint(root, step=10, world_size=2)

    def test_missing_rank_fails(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root)
            missing = (
                root
                / "global_step_10"
                / "critic"
                / "extra_state_world_size_2_rank_1.pt"
            )
            missing.unlink()
            with self.assertRaisesRegex(
                tool.CheckpointVerificationError, "missing critic/extra_state shards"
            ):
                tool.verify_checkpoint(root, step=10, world_size=2)

    def test_empty_shard_fails(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root)
            empty = root / "global_step_10" / "actor" / "model_world_size_2_rank_0.pt"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(
                tool.CheckpointVerificationError, "empty checkpoint shard"
            ):
                tool.verify_checkpoint(root, step=10, world_size=2)

    def test_unexpected_world_size_shard_fails(self):
        tool = _load_tool()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _make_checkpoint(root)
            unexpected = (
                root
                / "global_step_10"
                / "actor"
                / "optim_world_size_4_rank_2.pt"
            )
            unexpected.write_bytes(b"unexpected")
            with self.assertRaisesRegex(
                tool.CheckpointVerificationError, "unexpected actor/optim shards"
            ):
                tool.verify_checkpoint(root, step=10, world_size=2)


if __name__ == "__main__":
    unittest.main()
