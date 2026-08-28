from __future__ import annotations

import unittest

from agentmemorygym_verl.finalizer import _contiguous_positive_prefix


class FinalizerProgressCountsTest(unittest.TestCase):
    def test_counts_only_contiguous_observed_prefix(self) -> None:
        self.assertEqual(_contiguous_positive_prefix([]), 0)
        self.assertEqual(_contiguous_positive_prefix([1, 2, 3, 4]), 4)
        self.assertEqual(_contiguous_positive_prefix([0, 1, 2, 4, 100]), 2)
        self.assertEqual(_contiguous_positive_prefix([1, 1, 2, True, -3]), 2)


if __name__ == "__main__":
    unittest.main()
