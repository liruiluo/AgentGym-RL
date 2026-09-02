from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "async_plugins/scripts/restore_heldout_sources.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "camg_restore_heldout_sources_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=cwd, stderr=subprocess.STDOUT, text=True
    ).strip()


def make_repo(path: Path, name: str) -> str:
    path.mkdir()
    git("init", "-q", cwd=path)
    (path / f"{name}.txt").write_text(name + "\n", encoding="utf-8")
    git("add", ".", cwd=path)
    git(
        "-c",
        "user.name=CAMG Test",
        "-c",
        "user.email=camg-test@example.invalid",
        "commit",
        "-q",
        "-m",
        name,
        cwd=path,
    )
    return git("rev-parse", "HEAD", cwd=path)


class RestoreHeldoutSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()

    def fixture(self, root: Path):
        repositories = root / "repositories"
        repositories.mkdir()
        inner = repositories / "inner"
        commits = {"inner": make_repo(inner, "inner")}
        outer = repositories / "outer"
        make_repo(outer, "outer")
        git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(inner),
            "AgentGym",
            cwd=outer,
        )
        git(
            "-c",
            "user.name=CAMG Test",
            "-c",
            "user.email=camg-test@example.invalid",
            "commit",
            "-q",
            "-am",
            "bind inner",
            cwd=outer,
        )
        commits["outer"] = git("rev-parse", "HEAD", cwd=outer)
        lr = repositories / "literesearcher"
        commits["literesearcher_endpoint"] = make_repo(lr, "literesearcher")
        verl = repositories / "verl"
        commits["verl"] = make_repo(verl, "verl")

        bundles = root / "bundles"
        bundles.mkdir()
        mapping = {
            "outer": outer,
            "inner": inner,
            "literesearcher_endpoint": lr,
            "verl": verl,
        }
        paths = {}
        for label, repository in mapping.items():
            bundle = bundles / f"{label}.bundle"
            git("bundle", "create", str(bundle), "--all", cwd=repository)
            paths[label] = bundle
        specs = (
            self.tool.SourceSpec("outer", paths["outer"], commits["outer"], Path("AgentGym-RL")),
            self.tool.SourceSpec(
                "inner", paths["inner"], commits["inner"], Path("AgentGym-RL/AgentGym")
            ),
            self.tool.SourceSpec(
                "literesearcher_endpoint",
                paths["literesearcher_endpoint"],
                commits["literesearcher_endpoint"],
                Path("LiteResearcher-endpoint"),
            ),
            self.tool.SourceSpec("verl", paths["verl"], commits["verl"], Path("verl")),
        )
        return specs

    def test_restores_nested_sources_atomically_and_reuses_exact_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.fixture(root)
            target = root / "restored"
            created = self.tool.restore_sources(target, specs)
            reused = self.tool.restore_sources(target, specs)
            self.assertEqual(created["publication"], "created")
            self.assertEqual(reused["publication"], "reused_verified")
            self.assertEqual(
                git("rev-parse", "HEAD", cwd=target / "AgentGym-RL/AgentGym"),
                specs[1].commit,
            )

    def test_dirty_existing_checkout_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = self.fixture(root)
            target = root / "restored"
            self.tool.restore_sources(target, specs)
            (target / "verl/untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(self.tool.RestoreError, "checkout is dirty"):
                self.tool.restore_sources(target, specs)

    def test_invalid_commit_is_rejected_before_clone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = list(self.fixture(root))
            specs[0] = self.tool.SourceSpec(
                "outer", specs[0].bundle, "not-a-commit", specs[0].relative_destination
            )
            with self.assertRaisesRegex(self.tool.RestoreError, "full lowercase"):
                self.tool.restore_sources(root / "restored", tuple(specs))


if __name__ == "__main__":
    unittest.main()
