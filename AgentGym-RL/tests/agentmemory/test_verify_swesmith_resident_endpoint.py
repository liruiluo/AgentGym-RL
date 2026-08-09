#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agentmemory" / "verify_swesmith_resident_endpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_swesmith_resident_endpoint", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load SWE-smith endpoint verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SwesmithResidentEndpointProbeIndicesTests(unittest.TestCase):
    def test_accepts_eight_distinct_indices_across_the_curriculum(self) -> None:
        module = load_module()
        self.assertEqual(
            module.parse_probe_indices("0,1,13,14,30,31,47,48"),
            [0, 1, 13, 14, 30, 31, 47, 48],
        )

    def test_rejects_the_wrong_number_of_indices(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            module.parse_probe_indices("0,1,2,3,4,5,6")

    def test_rejects_duplicate_indices(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "distinct"):
            module.parse_probe_indices("0,1,2,3,4,5,6,6")

    def test_rejects_negative_indices(self) -> None:
        module = load_module()
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            module.parse_probe_indices("0,1,2,3,4,5,6,-1")


if __name__ == "__main__":
    unittest.main()
