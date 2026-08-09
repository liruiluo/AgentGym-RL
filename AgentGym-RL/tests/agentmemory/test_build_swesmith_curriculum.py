import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "agentmemory" / "build_swesmith_curriculum.py"
SPEC = importlib.util.spec_from_file_location("build_swesmith_curriculum", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _patch(path, before="old", after="new"):
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{before}\n"
        f"+{after}\n"
    )


def _row(instance_id, repo, path="src/code.py", changed="new", *, f2p=1, p2p=1):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "image_name": f"image/{repo}",
        "problem_statement": f"Fix {instance_id}",
        "patch": _patch(path, after=changed),
        "FAIL_TO_PASS": [f"f2p-{index}" for index in range(f2p)],
        "PASS_TO_PASS": [f"p2p-{index}" for index in range(p2p)],
    }


class BuildSwesmithCurriculumTest(unittest.TestCase):
    def _fixture(self, root, role="train"):
        rows = [
            _row("repo-a.small", "repo-a", changed="a"),
            _row("repo-a.large", "repo-a", changed="a\n+extra"),
            _row("repo-a.test", "repo-a", path="tests/test_code.py"),
            _row("repo-b.small", "repo-b", changed="b"),
            _row("repo-b.large", "repo-b", changed="b\n+extra"),
            _row("repo-b.multi", "repo-b"),
            _row("other.row", "other"),
        ]
        rows[5]["patch"] += _patch("src/second.py")
        shard = root / "shard.jsonl"
        shard_raw = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
        )
        shard.write_bytes(shard_raw)
        selected_ids = [row["instance_id"] for row in rows[:6]]
        selection = root / "base_ids.json"
        selection_raw = (json.dumps(selected_ids, indent=2) + "\n").encode("utf-8")
        selection.write_bytes(selection_raw)
        manifest = {
            "dataset_id": "fixture",
            "schema_version": "swesmith_jsonl_manifest_v1",
            "role": role,
            "selection": {
                "count": len(selected_ids),
                "mode": "instance_ids",
                "path": selection.name,
                "sha256": hashlib.sha256(selection_raw).hexdigest(),
                "split_contract": "fixture_repo_split",
            },
            "shards": [{
                "path": shard.name,
                "physical_rows": len(rows),
                "usable_rows": len(rows),
                "sha256": hashlib.sha256(shard_raw).hexdigest(),
            }],
            "source_corpus": {"root": "."},
            "upstream": {
                "repository": "SWE-bench/SWE-smith",
                "revision": "a" * 40,
            },
        }
        manifest_path = root / "base.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_builds_balanced_split_safe_curriculum_in_scan_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            output = root / "curriculum"
            report = MODULE.build_curriculum(
                base_manifest_path=manifest,
                output_dir=output,
                name="train2",
                expected_role="train",
                repositories=("repo-b", "repo-a"),
                per_repo=1,
                repository_quotas=None,
                max_changed_lines=4,
                max_f2p=2,
                min_p2p=1,
                max_problem_chars=3000,
                allowed_suffixes=(".py",),
                exclude_ids=set(),
            )

            self.assertEqual(report["count"], 2)
            self.assertEqual(
                [record["instance_id"] for record in report["records"]],
                ["repo-a.small", "repo-b.small"],
            )
            self.assertEqual(report["eligible_counts"], {"repo-b": 2, "repo-a": 2})
            selection = json.loads((output / "train2.instance_ids.json").read_text())
            self.assertEqual(selection, ["repo-a.small", "repo-b.small"])
            routing = [
                json.loads(line)
                for line in (output / "train2.routing.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["data_idx"] for row in routing], [0, 1])
            generated_manifest = json.loads((output / "train2.manifest.json").read_text())
            self.assertEqual(generated_manifest["role"], "train")
            self.assertEqual(generated_manifest["selection"]["count"], 2)
            self.assertTrue((output / generated_manifest["shards"][0]["path"]).is_file())

    def test_rejects_repository_outside_frozen_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "absent from the frozen split"):
                MODULE.build_curriculum(
                    base_manifest_path=manifest,
                    output_dir=root / "out",
                    name="invalid",
                    expected_role="train",
                    repositories=("other",),
                    per_repo=1,
                    repository_quotas=None,
                    max_changed_lines=4,
                    max_f2p=2,
                    min_p2p=1,
                    max_problem_chars=3000,
                    allowed_suffixes=(".py",),
                    exclude_ids=set(),
                )

    def test_rejects_role_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root, role="heldout")
            with self.assertRaisesRegex(ValueError, "expected role"):
                MODULE.build_curriculum(
                    base_manifest_path=manifest,
                    output_dir=root / "out",
                    name="invalid",
                    expected_role="train",
                    repositories=("repo-a",),
                    per_repo=1,
                    repository_quotas=None,
                    max_changed_lines=4,
                    max_f2p=2,
                    min_p2p=1,
                    max_problem_chars=3000,
                    allowed_suffixes=(".py",),
                    exclude_ids=set(),
                )


if __name__ == "__main__":
    unittest.main()
