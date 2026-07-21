from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTMEMORY_PACKAGE = ROOT.parent / "AgentGym/agentenv-agentmemory"
if str(AGENTMEMORY_PACKAGE) not in sys.path:
    sys.path.insert(0, str(AGENTMEMORY_PACKAGE))

from agentenv_agentmemory.domains.browsecomp import BrowseCompPlusFactory
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper

FORMAL_SOURCE = ROOT / "verl/utils/agentgym/formal_domain_v3.py"
FORMAL_SPEC = importlib.util.spec_from_file_location(
    "formal_domain_v3_for_browsecomp_test",
    FORMAL_SOURCE,
)
FORMAL_MODULE = importlib.util.module_from_spec(FORMAL_SPEC)
assert FORMAL_SPEC.loader is not None
FORMAL_SPEC.loader.exec_module(FORMAL_MODULE)


def generation_record():
    return {
        "response_token_count": 4,
        "max_response_tokens": 128,
        "finish_reason": "stop",
        "finish_reason_source": "vllm",
        "stop_reason": None,
        "backend_source": "vllm",
        "configured_eos_token_ids": [1],
        "tokenizer_pad_token_id": 0,
        "token_ids_are_exact": True,
        "backend_token_ids_are_exact": True,
        "truncated": False,
    }


class BrowseCompFormalDomainV3Test(unittest.TestCase):
    def test_native_search_transition_builds_a_formal_v3_row(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            ground_truth = root / "ground_truth.jsonl"
            decomposition = root / "decomposition.jsonl"
            ground_truth.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "final query",
                        "answer": "private final answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            decomposition.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": ["subquery", "final query"],
                        "answer": ["private sub answer", "private final answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            factory = BrowseCompPlusFactory(
                ground_truth_path=ground_truth,
                decomposition_path=decomposition,
                search_tool=lambda query: json.dumps(
                    [{"docid": "D1", "score": 1.0, "snippet": query}]
                ),
                judge=lambda question, predicted, correct: {
                    "correct": predicted == correct,
                    "confidence": 100.0,
                    "parse_error": False,
                },
            )
            wrapper = DomainEnvWrapper(factory)
            created = wrapper.create()
            raw = 'Action: search {"query": "evidence"}'
            stepped = wrapper.step(created["id"], raw)
            record = FORMAL_MODULE.build_formal_domain_step_v3(
                content=raw,
                score=stepped["reward"],
                task_round=1,
                done=stepped["done"],
                item_id="q1",
                parent_index=0,
                parent_group_uid="parent-q1",
                replica_index=0,
                trajectory_uid="trajectory-q1",
                exact_state_uid="state-q1",
                prompt_token_ids=[10, 11],
                response_token_ids=[20, 21, 22, 23],
                latest_observation=created["observation"],
                visible_prompt=created["observation"],
                generation_record=generation_record(),
                env_info_before=created["info"],
                env_info_after=stepped["info"],
            )
            wrapper.close(created["id"])

        self.assertEqual(record["domain_id"], "browsecomp_plus")
        self.assertEqual(record["action_execution"]["op"], "search")
        self.assertEqual(record["tool_ops"][0]["retrieved_docids"], ["D1"])
        self.assertEqual(record["reward_components"][0]["value"], 0.0)
        self.assertFalse(record["done"])


if __name__ == "__main__":
    unittest.main()
