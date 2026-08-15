import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "agentmemory"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_swesmith_formal_schedule",
    SCRIPTS / "build_swesmith_formal_schedule.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SwesmithScheduleOrderTest(unittest.TestCase):
    def _fixture(self, root):
        source = root / "source"
        source.mkdir()
        ids = [
            "alpha__one.aaaa1111.case0",
            "alpha__one.aaaa1111.case1",
            "beta__two.bbbb2222.case0",
            "beta__two.bbbb2222.case1",
            "gamma__three.cccc3333.case0",
            "gamma__three.cccc3333.case1",
            "delta__four.dddd4444.case0",
            "delta__four.dddd4444.case1",
        ]
        selection = source / "formal100.instance_ids.json"
        selection.write_text(json.dumps(ids, indent=2) + "\n")
        endpoint = source / "formal100.endpoint-index-map.jsonl"
        endpoint.write_text(
            "".join(
                json.dumps({"endpoint_data_idx": i, "instance_id": value}) + "\n"
                for i, value in enumerate(ids)
            )
        )
        source_order = [3, 0, 7, 2, 1, 4, 6, 5, 3, 0]
        routing = source / "formal100.routing.jsonl"
        routing.write_text(
            "".join(
                json.dumps(
                    {
                        "data_idx": value,
                        "extra_info": {
                            "index": value,
                            "instance_id_sha256": hashlib.sha256(
                                ids[value].encode()
                            ).hexdigest(),
                            "schedule_position": position,
                        },
                        "item_id": f"swesmith_{position}",
                    }
                )
                + "\n"
                for position, value in enumerate(source_order)
            )
        )
        image_manifest = source / "swesmith-image-manifest.json"
        images = []
        for owner, repository, revision in [
            ("alpha", "one", "aaaa1111"),
            ("beta", "two", "bbbb2222"),
            ("gamma", "three", "cccc3333"),
            ("delta", "four", "dddd4444"),
        ]:
            images.append(
                {"image": f"swebench/swesmith.x86_64.{owner}_1776_{repository}.{revision}"}
            )
        image_manifest.write_text(json.dumps({"images": images}) + "\n")
        manifest = {
            "dataset_id": "fixture",
            "role": "train",
            "schema_version": "swesmith_jsonl_manifest_v1",
            "selection": {
                "count": len(ids),
                "mode": "instance_ids",
                "path": selection.name,
                "sha256": _sha(selection),
                "split_contract": "fixture_random_order",
            },
            "shards": [],
        }
        (source / "formal100.manifest.json").write_text(json.dumps(manifest) + "\n")
        return source, ids, source_order

    def test_global_shuffle_replays_seed_and_repeats_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, ids, _ = self._fixture(root)
            output = root / "global"
            certificate = MODULE.build_schedule(
                source_dir=source,
                output_dir=output,
                mode="global-shuffle",
                seed=20260812,
                target_rows=10,
                batch_size=1,
                composition_block_updates=5,
                exclude_instance_ids=set(),
                minimum_repositories_per_block=2,
                maximum_repository_share=0.8,
            )
            self.assertEqual(certificate["status"], "pass")
            rows = [json.loads(line) for line in (output / "formal100.routing.jsonl").read_text().splitlines()]
            indices = [row["data_idx"] for row in rows]
            self.assertEqual(indices[:8], MODULE.seeded_permutation(8, 20260812))
            self.assertEqual(indices[8:], indices[:2])
            self.assertEqual(json.loads((output / "formal100.instance_ids.json").read_text()), ids)
            pairs = (output / "schedule-exact-repeat-pairs.jsonl").read_text().splitlines()
            self.assertEqual(len(pairs), 2)

    def test_exclude_only_filters_source_order_before_dense_reindex(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, ids, source_order = self._fixture(root)
            output = root / "filtered"
            excluded = {ids[0]}
            certificate = MODULE.build_schedule(
                source_dir=source,
                output_dir=output,
                mode="filter-source",
                seed=None,
                target_rows=10,
                batch_size=1,
                composition_block_updates=5,
                exclude_instance_ids=excluded,
                minimum_repositories_per_block=2,
                maximum_repository_share=0.8,
            )
            self.assertEqual(certificate["status"], "pass")
            new_ids = json.loads((output / "formal100.instance_ids.json").read_text())
            new_index = {value: index for index, value in enumerate(new_ids)}
            expected = [new_index[ids[index]] for index in source_order if ids[index] not in excluded]
            expected += expected[: 10 - len(expected)]
            rows = [json.loads(line) for line in (output / "formal100.routing.jsonl").read_text().splitlines()]
            self.assertEqual([row["data_idx"] for row in rows], expected)

    def test_certificate_rejects_seed_contract_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, _ = self._fixture(root)
            output = root / "global"
            MODULE.build_schedule(
                source_dir=source,
                output_dir=output,
                mode="global-shuffle",
                seed=20260812,
                target_rows=10,
                batch_size=1,
                composition_block_updates=5,
                exclude_instance_ids=set(),
                minimum_repositories_per_block=2,
                maximum_repository_share=0.8,
            )
            manifest_path = output / "formal100.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schedule"]["seed"] = 7
            manifest_path.write_text(json.dumps(manifest) + "\n")
            with self.assertRaisesRegex(ValueError, "declared shuffle seed"):
                MODULE.certify_schedule(output)


if __name__ == "__main__":
    unittest.main()
