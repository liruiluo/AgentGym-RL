from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCHEMAS_PATH = Path(__file__).resolve().parents[2] / "verl" / "workers" / "rollout" / "schemas.py"
FORMAL_DOMAIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl"
    / "utils"
    / "agentgym"
    / "formal_domain_v3.py"
)
PROMPT_ATTESTATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "agentmemory"
    / "attest_effective_memory_prompt.py"
)


def extract_static_string_assignments() -> dict[str, str]:
    tree = ast.parse(SCHEMAS_PATH.read_text(encoding="utf-8"))
    values: dict[str, str] = {}

    def resolve(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        raise ValueError(f"unsupported static string expression: {ast.dump(node)}")

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = resolve(node.value)
        except ValueError:
            continue
    return values


def load_schemas_module():
    module_name = "agentmemory_prompt_schema_for_test"
    spec = importlib.util.spec_from_file_location(module_name, SCHEMAS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PreTrainedTokenizer = object
    with patch.dict(
        sys.modules,
        {
            module_name: module,
            "torch": torch_stub,
            "transformers": transformers_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


def load_standalone_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingTokenizer:
    def __init__(self) -> None:
        self.conversations = None
        self.kwargs = None

    def apply_chat_template(self, conversations, **kwargs):
        self.conversations = conversations
        self.kwargs = kwargs
        return [101, 102, 103]


class FormalPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        values = extract_static_string_assignments()
        self.no_thinking_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT"]
        self.thinking_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING"]
        self.reasoning_prompt = values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING"]
        self.inventory_prompt = values[
            "AGENTMEMORY_ACTION_SYSTEM_PROMPT_LTM_KEY_INVENTORY"
        ]
        self.neutral_prompts = (
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL"],
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL"],
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL"],
        )
        self.neutral_inventory_prompt = values[
            "AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_LTM_KEY_INVENTORY"
        ]
        self.neutral_horizon_prompts = (
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON"],
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON"],
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON"],
        )
        self.neutral_horizon_inventory_prompt = values[
            "AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_LTM_KEY_INVENTORY"
        ]
        self.responsibility = values[
            "_AGENTMEMORY_CROSS_SESSION_MEMORY_RESPONSIBILITY"
        ]
        self.neutral_horizon_responsibility_prompts = (
            values[
                "AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY"
            ],
            values[
                "AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_NEUTRAL_HORIZON_RESPONSIBILITY"
            ],
            values[
                "AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_NEUTRAL_HORIZON_RESPONSIBILITY"
            ],
        )
        self.neutral_horizon_responsibility_inventory_prompt = values[
            "AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY_LTM_KEY_INVENTORY"
        ]
        self.latent_preference_prompts = (
            values["AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP"],
            values[
                "AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING_LATENT_PREFERENCE_SOP"
            ],
            values[
                "AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING_LATENT_PREFERENCE_SOP"
            ],
        )
        self.latent_preference_inventory_prompt = values[
            "AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP_LTM_KEY_INVENTORY"
        ]

    def test_both_prompts_have_native_action_contract(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for fragment in (
                "native WebShop bundled-shopping environment",
                "search[keywords]",
                "click[value]",
                "click[Buy Now]",
                "ADD requires key:string",
                "RETRIEVE accepts exactly one lookup field",
                "memory_id:string for exact readback",
                "Current-session trace clears",
                "Long-term memory persists across shopping sessions",
            ):
                self.assertIn(fragment, prompt)

    def test_both_prompts_have_explicit_memory_lifecycle(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for fragment in (
                "use ADD before click[Buy Now]",
                "At the start of every later shopping session",
                "use RETRIEVE",
                "does not reject an otherwise correct purchase when ADD was skipped",
            ):
                self.assertIn(fragment, prompt)

    def test_procedural_formal_resolver_renders_attested_lifecycle_prompt(self) -> None:
        schemas = load_schemas_module()
        formal_domain = load_standalone_module(
            "agentmemory_formal_domain_for_prompt_test",
            FORMAL_DOMAIN_PATH,
        )
        attestation = load_standalone_module(
            "agentmemory_prompt_attestation_for_integration_test",
            PROMPT_ATTESTATION_PATH,
        )
        with patch.dict(os.environ, {}, clear=True):
            expected_prompt = schemas.agentmemory_action_system_prompt()
            schema, resolved_prompt, source = (
                formal_domain.resolve_formal_runtime_contract(
                    {
                        "surface": formal_domain.FORMAL_WEBSHOP_PROCEDURAL_SURFACE_V2,
                        "memory_prompt_mode": "legacy",
                    },
                    webshop_v2_system_prompt=expected_prompt,
                )
            )

            handler = schemas.RolloutHandler(
                messages=[schemas.Message(role="user", content="RESET OBSERVATION")],
                task_name="agentmemory",
                item_id=0,
                score=0.0,
                done=False,
                input_ids=[],
                prompt_ids=[],
                response_ids=[],
                attention_mask=[],
                prompt_attention_mask=[],
                response_attention_mask=[],
                position_ids=[],
                prompt_position_ids=[],
                response_position_ids=[],
                loss_mask=[],
                prompt_loss_mask=[],
                response_loss_mask=[],
            )
            tokenizer = RecordingTokenizer()
            rendered_ids = handler.get_latest_observation_prompt(
                tokenizer,
                system_prompt=resolved_prompt,
            )
            receipt = attestation.build_attestation(
                prompt=resolved_prompt,
                memory_prompt_mode="legacy",
                ltm_inventory_mode="hidden",
                thinking_enabled=False,
                reasoning_enabled=False,
                require_lifecycle_sop=True,
            )

        self.assertEqual(schema, formal_domain.FORMAL_WEBSHOP_SCHEMA_V2)
        self.assertEqual(source, "rollout_webshop_v2")
        self.assertEqual(resolved_prompt, expected_prompt)
        self.assertEqual(rendered_ids, [101, 102, 103])
        self.assertEqual(
            tokenizer.conversations,
            [
                {"role": "system", "content": expected_prompt},
                {"role": "user", "content": "RESET OBSERVATION"},
            ],
        )
        self.assertTrue(receipt["lifecycle_sop_present"])
        self.assertEqual(receipt["missing_lifecycle_sop_fragments"], [])
        self.assertEqual(
            receipt["system_prompt_sha256"],
            hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest(),
        )
        self.assertIn("memory_id:string for exact readback", resolved_prompt)

    def test_neutral_prompts_keep_generic_tool_contract_without_timing_sop(self) -> None:
        forbidden = (
            "use ADD before click[Buy Now]",
            "At the start of every later shopping session",
            "before choosing a compatible product",
        )
        required = (
            "search[keywords]",
            "click[Buy Now]",
            "ADD requires key:string",
            "RETRIEVE accepts exactly one lookup field",
            "memory_id:string for exact readback",
            "reads only text you previously wrote to long-term memory",
            "Long-term memory persists across shopping sessions",
            "remains hidden until RETRIEVE exposes it",
        )
        for prompt in self.neutral_prompts:
            for fragment in required:
                self.assertIn(fragment, prompt)
            for fragment in forbidden:
                self.assertNotIn(fragment, prompt)

    def test_neutral_key_inventory_keeps_values_hidden(self) -> None:
        self.assertIn("key-only long-term memory inventory", self.neutral_inventory_prompt)
        self.assertIn("values remain hidden until RETRIEVE", self.neutral_inventory_prompt)
        self.assertNotIn("use ADD before click[Buy Now]", self.neutral_inventory_prompt)

    def test_neutral_horizon_adds_only_objective_task_scope(self) -> None:
        horizon = (
            "This episode has six sequential shopping sessions. Later-session "
            "compatibility constraints may refer to products purchased in earlier "
            "sessions."
        )
        forbidden = (
            "use ADD before click[Buy Now]",
            "At the start of every later shopping session",
            "before choosing a compatible product",
        )
        for neutral, horizon_prompt in zip(
            self.neutral_prompts,
            self.neutral_horizon_prompts,
        ):
            self.assertNotIn(horizon, neutral)
            self.assertEqual(horizon_prompt, neutral + " " + horizon)
            for fragment in forbidden:
                self.assertNotIn(fragment, horizon_prompt)
        self.assertIn(
            "key-only long-term memory inventory",
            self.neutral_horizon_inventory_prompt,
        )
        self.assertIn(horizon, self.neutral_horizon_inventory_prompt)

    def test_neutral_horizon_responsibility_adds_one_non_sop_sentence(self) -> None:
        self.assertEqual(
            self.responsibility,
            "Across shopping sessions, you are responsible for preserving and "
            "accessing any facts needed for later decisions.",
        )
        for horizon_prompt, responsibility_prompt in zip(
            self.neutral_horizon_prompts,
            self.neutral_horizon_responsibility_prompts,
        ):
            self.assertEqual(
                responsibility_prompt,
                horizon_prompt + " " + self.responsibility,
            )
            self.assertEqual(responsibility_prompt.count(self.responsibility), 1)
            self.assertNotIn("use ADD before click[Buy Now]", responsibility_prompt)
            self.assertNotIn(
                "At the start of every later shopping session",
                responsibility_prompt,
            )
        self.assertIn(
            "key-only long-term memory inventory",
            self.neutral_horizon_responsibility_inventory_prompt,
        )
        self.assertEqual(
            self.neutral_horizon_responsibility_inventory_prompt.count(
                self.responsibility
            ),
            1,
        )

    def test_latent_preference_sop_preserves_evidence_and_infers_without_fixed_shots(
        self,
    ) -> None:
        required = (
            "confirmed choice as preference evidence",
            "customer-profile memory",
            "preference axis",
            "inferred value",
            "Do not assume a fixed number",
            "use ADD before click[Buy Now]",
            "use UPDATE",
            "At the start of every later shopping session",
            "use RETRIEVE",
            "later application sessions",
            "memory_id:string for exact readback",
        )
        forbidden = (
            "save one concise memory containing that product's identity",
            "attributes needed for later compatibility decisions",
            "exactly two examples",
            "exactly three examples",
        )
        for prompt in self.latent_preference_prompts:
            for fragment in required:
                self.assertIn(fragment, prompt)
            for fragment in forbidden:
                self.assertNotIn(fragment, prompt)
        self.assertIn(
            "key-only long-term memory inventory",
            self.latent_preference_inventory_prompt,
        )
        self.assertIn(
            "values remain hidden until RETRIEVE",
            self.latent_preference_inventory_prompt,
        )

    def test_prompt_mode_is_opt_in_and_validated(self) -> None:
        module = load_schemas_module()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                module.agentmemory_action_system_prompt(),
                module.AGENTMEMORY_ACTION_SYSTEM_PROMPT,
            )
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_PROMPT_MODE": "neutral"},
            clear=True,
        ):
            self.assertEqual(
                module.agentmemory_action_system_prompt(),
                module.AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL,
            )
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_PROMPT_MODE": "neutral_horizon"},
            clear=True,
        ):
            self.assertEqual(
                module.agentmemory_action_system_prompt(),
                module.AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON,
            )
        with patch.dict(
            os.environ,
            {
                "AGENTMEMORY_MEMORY_PROMPT_MODE": (
                    "neutral_horizon_responsibility"
                )
            },
            clear=True,
        ):
            self.assertEqual(
                module.agentmemory_action_system_prompt(),
                module.AGENTMEMORY_ACTION_SYSTEM_PROMPT_NEUTRAL_HORIZON_RESPONSIBILITY,
            )
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_PROMPT_MODE": "latent_preference_sop"},
            clear=True,
        ):
            self.assertEqual(
                module.agentmemory_action_system_prompt(),
                module.AGENTMEMORY_ACTION_SYSTEM_PROMPT_LATENT_PREFERENCE_SOP,
            )
        with patch.dict(
            os.environ,
            {
                "AGENTMEMORY_MEMORY_PROMPT_MODE": "selective_memory_sop",
                "AGENTMEMORY_SURFACE": (
                    "agentmemory_webshop_selective_memory_use_top1_train_v1"
                ),
            },
            clear=True,
        ):
            prompt = module.agentmemory_action_system_prompt(
                surface=os.environ["AGENTMEMORY_SURFACE"]
            )
            for fragment in (
                "First decide whether the current request already states every attribute",
                "explicit current requirements override profile history",
                "should not ADD or RETRIEVE merely by habit",
                "use RETRIEVE to expose the saved current profile",
                "RETRIEVE requires exactly query:string",
                "memory_id and top_k are forbidden",
            ):
                self.assertIn(fragment, prompt)
            self.assertNotIn("use ADD before click[Buy Now]", prompt)
            self.assertNotIn("confirmed choice as preference evidence", prompt)
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_MEMORY_PROMPT_MODE": "instruction"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "MEMORY_PROMPT_MODE"):
                module.agentmemory_action_system_prompt()

    def test_natural_filesystem_prompt_exposes_only_codex_workspace_tools(self) -> None:
        module = load_schemas_module()
        prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.NATURAL_FILESYSTEM_SURFACE,
        )
        for fragment in (
            "native WebShop bundled-shopping environment",
            "search[keywords]",
            "click[Buy Now]",
            "click[its displayed ASIN]",
            "after the exact product page opens",
            'shell_command {"command":"rg -n pattern .","workdir":".","timeout_ms":10000}',
            "apply_patch is followed on the next line",
            "ordinary file utilities and pinned rg",
            "has no network",
            "workspace persists across shopping sessions within this episode",
            "Workspace actions have zero task reward",
            "no host-path access and no dedicated memory API",
            "workspace starts empty and contains only files that you create",
            "Before an action that advances to a new shopping session",
            "Copy the exact policy-visible field name and value needed later",
            "every Add File content line must begin with `+`",
            "latest-observation-only input",
            "first action must use shell_command to discover and print saved Confirmed lines",
            "rg --hidden -n '^Confirmed ' .",
            "`rg` without `--hidden` is insufficient",
            "If stdout is empty, retry the exact `rg --hidden` command",
            "do not guess a path",
            "until the preceding session's note contents have appeared in shell output",
            "Never infer or recreate the missing value from the choice table",
            "Listing a directory or reading a different file is not reading the note",
            "copy the chosen approved card's complete Product title into search[...] "
            "without shortening it",
            "Never invent a `Confirmed ... to buy:` field",
            "After an Add File action returns Done!, keep that successful note unchanged",
            "new note, use Add File with a new path",
            "never append Result or feedback text to the action",
            "separate turns and must never be emitted together",
            "Earlier turn (complete reply):\napply_patch",
            "*** Add File: .agent_memory/example.md\n+service port: 4317",
            "Later turn (complete reply):\n"
            "shell_command {\"command\":\"rg --hidden -n '^service port:' .\"",
        ):
            self.assertIn(fragment, prompt)
        self.assertLess(
            prompt.index("Earlier turn (complete reply):"),
            prompt.index("Later turn (complete reply):"),
        )
        self.assertNotIn("Paula Deen", prompt)
        self.assertNotIn("Betty Crocker", prompt)
        for forbidden in (
            'Read {"path"',
            'Write {"path"',
            'Edit {"path"',
            'Grep {"pattern"',
            'Glob {"pattern"',
            "ADD requires",
            "RETRIEVE accepts",
            "use ADD",
            "use RETRIEVE",
            "memory_id",
            "Long-term memory",
        ):
            self.assertNotIn(forbidden, prompt)

        with self.assertRaisesRegex(ValueError, "ltm_inventory_mode"):
            module.agentmemory_action_system_prompt(
                ltm_inventory_mode="keys",
                memory_prompt_mode="natural_filesystem",
                surface=module.NATURAL_FILESYSTEM_SURFACE,
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            module.agentmemory_action_system_prompt(
                memory_prompt_mode="natural_filesystem",
                surface="agentmemory_webshop_procedural_natural_chain_train_v1",
            )

    def test_no_workspace_intervention_prompt_exposes_only_native_actions(self) -> None:
        module = load_schemas_module()
        prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.NATURAL_FILESYSTEM_SURFACE,
            workspace_enabled=False,
        )
        for fragment in (
            "without a persistent workspace",
            "exactly one executable native browser action",
            "search[keywords]",
            "click[Buy Now]",
            "shell_command and apply_patch are unavailable and invalid",
        ):
            self.assertIn(fragment, prompt)
        for forbidden in (
            "shell_command JSON action",
            "multiline apply_patch action",
            "workspace persists across shopping sessions",
            "Workspace actions have zero task reward",
            "workspace starts empty",
            "exact policy-visible field name and value",
            "preceding session's note contents",
            "Earlier turn (complete reply)",
            "+service port: 4317",
            "ADD requires",
            "RETRIEVE accepts",
        ):
            self.assertNotIn(forbidden, prompt)
        with self.assertRaisesRegex(ValueError, "only for natural_filesystem"):
            module.agentmemory_action_system_prompt(
                memory_prompt_mode="legacy",
                workspace_enabled=False,
            )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            module.agentmemory_action_system_prompt(
                memory_prompt_mode="natural_filesystem",
                surface=module.NATURAL_FILESYSTEM_SURFACE,
                workspace_enabled=0,
            )

    def test_recency_filesystem_prompt_requires_current_state_replacement(self) -> None:
        module = load_schemas_module()
        prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
        )
        for fragment in (
            "one ordinary file as the current confirmed user-preference record",
            "exact policy-visible record line beginning with `Current preference:`",
            "copy that exact line verbatim",
            "use apply_patch Update File on the existing current-state file",
            "new value replaces the old value",
            "do not leave conflicting current and stale values",
            "rg --hidden -n '^Current preference:' .",
            "first use shell_command to print the current preference record",
            "Do not infer the missing value from the choice table",
            "A merchandise category, product title, or option-table entry is not the preference record",
            "correct and retry the workspace read before any search or click",
            "use exactly five physical lines",
            "line 3 begins with `*** Add File: `, including the colon and following space",
            "line 4 begins with `+` immediately followed by the task's exact record",
            "Never combine two patch lines on one physical line",
            "use exactly seven physical lines",
            "line 3 begins with `*** Update File: `, including the colon and following space",
            "line 4 is exactly `@@`",
            "line 5 begins with `-` immediately followed by the exact old record",
            "line 6 begins with `+` immediately followed by the exact new task-provided record",
            "assume the workspace is unchanged and fix and retry the workspace action",
            "only `Done!` proves that the patch succeeded",
            "No concrete preference value or filename is demonstrated here",
        ):
            self.assertIn(fragment, prompt)
        for forbidden in (
            "After an Add File action returns Done!, keep that successful note unchanged",
            "ADD requires",
            "RETRIEVE accepts",
            "memory_id",
            "Long-term memory",
            "Current service region:",
            ".agent_memory/current.md",
            "Current preference: east",
            "Current preference: west",
        ):
            self.assertNotIn(forbidden, prompt)

        no_workspace = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
            workspace_enabled=False,
        )
        self.assertIn("without a persistent workspace", no_workspace)
        self.assertNotIn("current confirmed user-preference record", no_workspace)

    def test_distractor_filesystem_prompt_requires_seed_aware_search(self) -> None:
        module = load_schemas_module()
        prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
        )
        for fragment in (
            "workspace may start with harness-seeded ordinary profile notes",
            "background records, not policy actions, hidden answers",
            "do not rewrite or delete the seeded notes",
            "rg --hidden -n -i '(current|preference|profile)' .",
            "Ignore superseded history, another customer's record",
            "Do not rely on filenames, directory order, or the number of files",
            "exact policy-authored `Current preference:` line",
            "never infer the missing value from the current choice table",
            "Create the current record with exactly five physical lines",
            "Only `Done!` proves a patch succeeded",
        ):
            self.assertIn(fragment, prompt)
        for forbidden in (
            "workspace starts empty and contains only files that you create",
            "Current preference: east",
            "Current preference: west",
            "ADD requires",
            "RETRIEVE accepts",
        ):
            self.assertNotIn(forbidden, prompt)
        no_workspace = module.agentmemory_action_system_prompt(
            memory_prompt_mode="natural_filesystem",
            surface=module.DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
            workspace_enabled=False,
        )
        self.assertIn("without a persistent workspace", no_workspace)
        self.assertNotIn("harness-seeded ordinary profile notes", no_workspace)

    def test_all_filesystem_surfaces_have_surface_specific_prompts(self) -> None:
        module = load_schemas_module()
        expected_surfaces = {
            module.NATURAL_FILESYSTEM_SURFACE,
            module.LATENT_PREFERENCE_FILESYSTEM_SURFACE,
            module.RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
            module.DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
            module.COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
            module.INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
            module.SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
            module.NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
        }
        self.assertEqual(set(module.FILESYSTEM_SURFACES), expected_surfaces)
        cases = {
            module.LATENT_PREFERENCE_FILESYSTEM_SURFACE: (
                "confirmed choice as preference evidence",
                "customer-profile memory",
                "apply the retrieved preference in later application sessions",
            ),
            module.INTENT_CLARIFICATION_FILESYSTEM_SURFACE: (
                'ASK {"field":"..."}',
                "infer and fill the missing field",
                "CLARIFY observation",
            ),
            module.SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE: (
                "branch-conditioned ordinary profile file",
                "first decide whether the current request already states every attribute needed",
                "do not read the profile merely by habit",
            ),
        }
        prompts = {}
        for surface, fragments in cases.items():
            prompt = module.agentmemory_action_system_prompt(
                memory_prompt_mode="natural_filesystem",
                surface=surface,
            )
            prompts[surface] = prompt
            for fragment in fragments:
                self.assertIn(fragment, prompt)
            for forbidden in ("ADD requires", "RETRIEVE accepts", "memory_id:string"):
                self.assertNotIn(forbidden, prompt)
        self.assertNotIn(
            'ASK {"field":"color"}',
            prompts[module.INTENT_CLARIFICATION_FILESYSTEM_SURFACE],
        )
        self.assertEqual(len(set(prompts.values())), len(prompts))

    def test_surface_specific_top1_and_clarification_contracts(self) -> None:
        module = load_schemas_module()
        distractor_surface = (
            "agentmemory_webshop_distractor_robustness_top1_train_v1"
        )
        intent_surface = "agentmemory_webshop_intent_clarification_train_v1"
        distractor_prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="latent_preference_sop",
            surface=distractor_surface,
        )
        intent_prompt = module.agentmemory_action_system_prompt(
            memory_prompt_mode="latent_preference_sop",
            surface=intent_surface,
        )
        for prompt in (distractor_prompt, intent_prompt):
            self.assertIn("RETRIEVE requires exactly query:string", prompt)
            self.assertIn("one highest-ranked matching memory", prompt)
            self.assertIn("memory_id and top_k are forbidden", prompt)
            self.assertNotIn("optional top_k:int", prompt)
            self.assertNotIn("memory_id:string for exact readback", prompt)
        self.assertNotIn("ASK requires field:string", distractor_prompt)
        self.assertNotIn("CLARIFY observation", distractor_prompt)
        self.assertIn("ASK requires field:string", intent_prompt)
        self.assertIn("CLARIFY observation", intent_prompt)

    def test_action_listing_mode_is_opt_in_and_validated(self) -> None:
        module = load_schemas_module()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.agentmemory_action_listing_mode(), "separate")
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_ACTION_LISTING_MODE": "unified"},
            clear=True,
        ):
            self.assertEqual(module.agentmemory_action_listing_mode(), "unified")
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_ACTION_LISTING_MODE": "ranked"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "ACTION_LISTING_MODE"):
                module.agentmemory_action_listing_mode()

    def test_reply_rules_match_thinking_mode(self) -> None:
        self.assertIn("Output excludes", self.no_thinking_prompt)
        self.assertIn("<think> blocks", self.no_thinking_prompt)
        self.assertIn("You may first reason inside a single <think>", self.thinking_prompt)
        self.assertIn("After the closing </think>", self.thinking_prompt)
        self.assertIn("Write `Thought:` followed by brief free-form reasoning", self.reasoning_prompt)
        self.assertIn("after the final `Action:` label", self.reasoning_prompt)
        self.assertIn("PPO trains the complete sampled Thought-and-Action response", self.reasoning_prompt)
        self.assertIn("Output excludes markdown and <think> blocks", self.reasoning_prompt)

    def test_prompts_contain_no_removed_action(self) -> None:
        for prompt in (self.no_thinking_prompt, self.thinking_prompt, self.reasoning_prompt):
            for forbidden in (
                "GROUND",
                'SEARCH {"query"',
                'BUY {"product_id"',
                "PAGE accepts",
                "try a different candidate",
            ):
                self.assertNotIn(forbidden, prompt)

    def test_key_inventory_prompt_exposes_only_policy_authored_ids_and_keys(self) -> None:
        self.assertIn("key-only long-term memory inventory", self.inventory_prompt)
        self.assertIn("memory_id and a policy-authored lookup key", self.inventory_prompt)
        self.assertIn("at most 24", self.inventory_prompt)
        self.assertIn("put product identity and compatibility facts in value, not key", self.inventory_prompt)
        self.assertIn("values remain hidden until RETRIEVE", self.inventory_prompt)
        self.assertIn("RETRIEVE matches both the key and value", self.inventory_prompt)
        self.assertNotIn("target", self.inventory_prompt.lower())


if __name__ == "__main__":
    unittest.main()
