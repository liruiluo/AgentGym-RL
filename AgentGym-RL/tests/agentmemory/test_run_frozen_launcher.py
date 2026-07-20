from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FREEZER = ROOT / "scripts/run_frozen_launcher.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


class FrozenLauncherTests(unittest.TestCase):
    def test_in_place_source_mutation_cannot_change_running_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run with spaces"
            source = root / "source launcher.sh"
            log = root / "launcher.log"
            ready = root / "ready"
            release = root / "release"
            source.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
LOG=$1
READY=$2
RELEASE=$3
printf 'launcher_path=%s\\n' "$0" >> "$LOG"
printf 'frozen_flag=%s\\n' "${AGENTMEMORY_FROZEN_LAUNCHER:-}" >> "$LOG"
printf 'source_sha=%s\\n' "${AGENTMEMORY_LAUNCHER_SOURCE_SHA256:-}" >> "$LOG"
touch "$READY"
while [ ! -e "$RELEASE" ]; do sleep 0.02; done
printf 'ORIGINAL_TAIL\\n' >> "$LOG"
""",
                encoding="utf-8",
            )
            original_sha = _sha256(source)

            process = subprocess.Popen(
                [
                    "/usr/bin/env",
                    "bash",
                    str(FREEZER),
                    str(run_dir),
                    str(source),
                    str(log),
                    str(ready),
                    str(release),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists(), process.stderr.read() if process.poll() is not None else "")

            # Truncate and rewrite the same source path while its frozen copy is paused.
            source.write_text(
                "#!/usr/bin/env bash\nprintf 'MUTATED_TAIL\\n' >> \"$1\"\nexit 127\n",
                encoding="utf-8",
            )
            self.assertNotEqual(_sha256(source), original_sha)
            release.touch()
            stdout, stderr = process.communicate(timeout=5)

            self.assertEqual(process.returncode, 0, f"stdout={stdout}\nstderr={stderr}")
            output = log.read_text(encoding="utf-8")
            self.assertIn("ORIGINAL_TAIL", output)
            self.assertNotIn("MUTATED_TAIL", output)
            self.assertIn("frozen_flag=1", output)
            self.assertIn(f"source_sha={original_sha}", output)

            snapshots = list((run_dir / "launcher_snapshots").glob("*/manifest.env"))
            self.assertEqual(len(snapshots), 1)
            manifest = _read_manifest(snapshots[0])
            frozen = Path(manifest["frozen_path"])
            self.assertEqual(manifest["source_path"], str(source))
            self.assertEqual(manifest["source_sha256"], original_sha)
            self.assertEqual(manifest["frozen_sha256"], original_sha)
            self.assertEqual(_sha256(frozen), original_sha)
            self.assertIn(f"launcher_path={frozen}", output)
            self.assertIn(f"sha256={original_sha}", stderr)

    def test_missing_source_fails_before_creating_a_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            result = subprocess.run(
                [
                    "/usr/bin/env",
                    "bash",
                    str(FREEZER),
                    str(run_dir),
                    str(root / "missing.sh"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 70)
            self.assertIn("source launcher is not a regular file", result.stderr)
            self.assertFalse((run_dir / "launcher_snapshots").exists())


if __name__ == "__main__":
    unittest.main()
