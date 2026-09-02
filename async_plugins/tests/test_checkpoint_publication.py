from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentmemorygym_verl.checkpoint_publication import (
    MODEL_MANIFEST_SCHEMA,
    build_merged_hf_manifest,
    inspect_fsdp_actor_checkpoint,
)


COMMITS = {
    "outer_commit": "1" * 40,
    "inner_commit": "2" * 40,
    "verl_commit": "3" * 40,
}


def _make_checkpoint(root: Path, *, step: int = 400, world_size: int = 2) -> Path:
    run = root / "run"
    actor = run / f"checkpoints/global_step_{step}/actor"
    (actor / "huggingface").mkdir(parents=True)
    (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(
        f"{step}\n", encoding="utf-8"
    )
    (actor / "fsdp_config.json").write_text(
        json.dumps({"world_size": world_size}) + "\n", encoding="utf-8"
    )
    for rank in range(world_size):
        (actor / f"model_world_size_{world_size}_rank_{rank}.pt").write_bytes(
            f"rank={rank}".encode()
        )
    (actor / "huggingface/config.json").write_text("{}\n", encoding="utf-8")
    (actor / "huggingface/tokenizer.json").write_text("{}\n", encoding="utf-8")
    return run


def _make_model(root: Path) -> Path:
    model = root / "merged-hf"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return model


class CheckpointPublicationTest(unittest.TestCase):
    def test_inspects_exact_fsdp_actor_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            report = inspect_fsdp_actor_checkpoint(_make_checkpoint(Path(temp)))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checkpoint_step"], 400)
        self.assertEqual(report["world_size"], 2)
        self.assertEqual(len(report["model_shards"]), 2)
        self.assertEqual(len(report["huggingface_metadata"]), 2)

    def test_rejects_incomplete_or_wrong_endpoint_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            run = _make_checkpoint(Path(temp))
            (run / "checkpoints/global_step_400/actor/model_world_size_2_rank_1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "shard set mismatch"):
                inspect_fsdp_actor_checkpoint(run)

        with tempfile.TemporaryDirectory() as temp:
            run = _make_checkpoint(Path(temp))
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(
                "390\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "latest checkpoint is 390"):
                inspect_fsdp_actor_checkpoint(run)

    def test_builds_evaluator_compatible_exact_model_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            model = _make_model(Path(temp))
            manifest = build_merged_hf_manifest(
                model,
                training_run_id="amg_compactionrl_formal400",
                checkpoint_step=400,
                **COMMITS,
            )
        self.assertEqual(manifest["schema"], MODEL_MANIFEST_SCHEMA)
        self.assertEqual(manifest["checkpoint_step"], 400)
        self.assertEqual(manifest["source_commits"]["outer"], "1" * 40)
        self.assertEqual(len(manifest["files"]), 3)

    def test_rejects_symlink_empty_hidden_and_nonterminal_publications(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = _make_model(root)
            (model / "link").symlink_to(model / "config.json")
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_merged_hf_manifest(
                    model,
                    training_run_id="run",
                    checkpoint_step=400,
                    **COMMITS,
                )

        with tempfile.TemporaryDirectory() as temp:
            model = _make_model(Path(temp))
            with self.assertRaisesRegex(ValueError, "only at update400"):
                build_merged_hf_manifest(
                    model,
                    training_run_id="run",
                    checkpoint_step=399,
                    **COMMITS,
                )


if __name__ == "__main__":
    unittest.main()
