from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "probe_literesearcher_retrieval_memory.py"
WRAPPER_PATH = ROOT / "scripts" / "run_literesearcher_retrieval_memory_probe.sh"
SPEC = importlib.util.spec_from_file_location("retrieval_memory_probe", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_parity_separates_url_order_from_float_tolerance() -> None:
    expected = [["u1", 0.5], ["u2", 0.25], ["u3", 0.125]]
    actual = [["u1", 0.5000004], ["u3", 0.1250003], ["u2", 0.2500002]]

    parity = PROBE.compare_url_scores(expected, actual, score_atol=1e-5)

    assert parity["ordered_url_exact"] is False
    assert parity["top1_url_exact"] is True
    assert parity["topk_set_overlap_ratio"] == 1.0
    assert parity["common_url_scores_within_tolerance"] is True
    assert parity["ordered_url_score_within_tolerance"] is False


def test_parity_reports_partial_topk_overlap() -> None:
    expected = [[f"u{index}", float(index)] for index in range(5)]
    actual = [["u0", 0.0], ["u1", 1.0], ["u2", 2.0], ["u3", 3.0], ["other", 4.0]]

    parity = PROBE.compare_url_scores(expected, actual, score_atol=1e-5)

    assert parity["top1_url_exact"] is True
    assert parity["topk_set_overlap_count"] == 4
    assert parity["topk_set_overlap_ratio"] == 0.8


def test_wrapper_keeps_trap_owner_instead_of_execing_probe() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")
    assert "trap cleanup EXIT INT TERM HUP" in text
    assert '"$PYTHON_BIN" "$PROBE_SCRIPT" "$@" &' in text
    assert 'exec "$PYTHON_BIN"' not in text
    subprocess.run(["bash", "-n", str(WRAPPER_PATH)], check=True)
