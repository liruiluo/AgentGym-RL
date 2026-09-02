from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "async_plugins/scripts/publish_merged_checkpoint.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "camg_publish_merged_checkpoint_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_checkpoint(root: Path, *, step: int = 200, world_size: int = 2) -> None:
    (root / "latest_checkpointed_iteration.txt").write_text(
        f"{step}\n", encoding="utf-8"
    )
    step_root = root / f"global_step_{step}"
    step_root.mkdir(parents=True)
    (step_root / "data.pt").write_bytes(b"data")
    for role in ("actor", "critic"):
        role_root = step_root / role
        role_root.mkdir()
        (role_root / "fsdp_config.json").write_text(
            json.dumps({"world_size": world_size}), encoding="utf-8"
        )
        for stem in ("model", "optim", "extra_state"):
            for rank in range(world_size):
                (role_root / f"{stem}_world_size_{world_size}_rank_{rank}.pt").write_bytes(
                    f"{role}-{stem}-{rank}".encode("ascii")
                )


def fake_merge(command, **_kwargs):
    target = Path(command[command.index("--target_dir") + 1])
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}\n", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"weights")
    return mock.Mock(returncode=0)


class PublishMergedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()
        self.source_commits = {
            "outer": "1" * 40,
            "inner": "2" * 40,
            "verl": "3" * 40,
        }

    def kwargs(self, root: Path):
        checkpoint_root = root / "checkpoints"
        checkpoint_root.mkdir()
        make_checkpoint(checkpoint_root)
        verl_root = root / "verl"
        verl_root.mkdir()
        python = root / "python"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o700)
        return {
            "checkpoint_root": checkpoint_root,
            "checkpoint_step": 200,
            "world_size": 2,
            "model_path": root / "model",
            "manifest_path": root / "model-manifest.json",
            "merge_log_path": root / "merge.log",
            "training_run_id": "agemem-formal200",
            "source_commits": self.source_commits,
            "verl_root": verl_root,
            "python_executable": python,
        }

    def test_publish_then_verify_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(Path(directory))
            with mock.patch.object(
                self.tool.subprocess, "run", side_effect=fake_merge
            ) as merger:
                created = self.tool.publish_checkpoint(**kwargs)
                reused = self.tool.publish_checkpoint(**kwargs)
            self.assertEqual(created["publication"], "created")
            self.assertEqual(reused["publication"], "reused_verified")
            self.assertEqual(merger.call_count, 1)
            payload = json.loads(kwargs["manifest_path"].read_text())
            self.assertEqual(payload["schema"], self.tool.MODEL_MANIFEST_SCHEMA)
            self.assertEqual(payload["checkpoint_step"], 200)
            self.assertEqual(len(payload["files"]), 2)

    def test_partial_existing_publication_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(Path(directory))
            kwargs["model_path"].mkdir()
            with self.assertRaisesRegex(
                self.tool.PublicationError, "partial publication"
            ):
                self.tool.publish_checkpoint(**kwargs)

    def test_merge_log_cannot_create_the_model_destination_early(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(Path(directory))
            kwargs["merge_log_path"] = kwargs["model_path"] / "merge.log"
            with self.assertRaisesRegex(
                self.tool.PublicationError, "merge log must live outside"
            ):
                self.tool.publish_checkpoint(**kwargs)

    def test_missing_checkpoint_shard_fails_before_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(Path(directory))
            missing = (
                kwargs["checkpoint_root"]
                / "global_step_200/critic/optim_world_size_2_rank_1.pt"
            )
            missing.unlink()
            with mock.patch.object(self.tool.subprocess, "run") as merger:
                with self.assertRaisesRegex(
                    self.tool.PublicationError, "checkpoint verification failed"
                ):
                    self.tool.publish_checkpoint(**kwargs)
            merger.assert_not_called()

    def test_merge_output_symlink_fails_closed(self):
        def unsafe_merge(command, **_kwargs):
            target = Path(command[command.index("--target_dir") + 1])
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}\n", encoding="utf-8")
            (target / "model.safetensors").symlink_to(target / "config.json")
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            kwargs = self.kwargs(Path(directory))
            with mock.patch.object(
                self.tool.subprocess, "run", side_effect=unsafe_merge
            ):
                with self.assertRaisesRegex(
                    self.tool.PublicationError, "contains a symlink"
                ):
                    self.tool.publish_checkpoint(**kwargs)


if __name__ == "__main__":
    unittest.main()
