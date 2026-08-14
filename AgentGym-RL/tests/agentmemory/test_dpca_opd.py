import json
import tempfile
import unittest
from pathlib import Path

import torch

from verl.experimental.remote_opd.dpca import (
    DPCA_OPD_ADVANTAGES,
    DPCA_OPD_TOKEN_MASK,
    DPCAOPDError,
    DPCAOPDSettings,
    RemoteDPCAOPDScorer,
    attach_dpca_opd_advantages,
    dpca_minimal_chunks,
    dpca_semantic_prior_credit,
    parse_qwen_chatml_generation_prompt,
)
from verl.utils.agentgym.rollout_context import (
    AGENTMEMORY_ACTION_TEXT,
    AGENTMEMORY_GENERATION_PROMPT_DIGEST,
    AGENTMEMORY_GENERATION_PROMPT_LENGTH,
    AGENTMEMORY_GENERATION_RESPONSE_DIGEST,
    AGENTMEMORY_GENERATION_RESPONSE_LENGTH,
    AGENTMEMORY_PACKED_PROMPT_DIGEST,
    AGENTMEMORY_PACKED_PROMPT_LENGTH,
    AGENTMEMORY_PACKED_RESPONSE_DIGEST,
    AGENTMEMORY_PACKED_RESPONSE_LENGTH,
    AGENTMEMORY_STEP_RECORD_JSON,
    prompt_token_digest,
)


class PieceTokenizer:
    def __init__(self, pieces, *, special_ids=()):
        self.pieces = dict(pieces)
        self.all_special_ids = list(special_ids)

    def decode(self, ids, skip_special_tokens=False):
        special = set(self.all_special_ids)
        return "".join(
            ""
            if skip_special_tokens and int(token_id) in special
            else self.pieces[int(token_id)]
            for token_id in ids
        )


class CharacterTeacherTokenizer:
    eos_token = "¤"
    eos_token_id = ord(eos_token)
    im_end_token = "<|im_end|>"
    im_end_token_id = 1_000_001
    unk_token_id = -1
    all_special_ids = [eos_token_id, im_end_token_id]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        position = 0
        while position < len(text):
            if text.startswith(self.im_end_token, position):
                ids.append(self.im_end_token_id)
                position += len(self.im_end_token)
            else:
                ids.append(ord(text[position]))
                position += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.im_end_token_id:
                piece = self.im_end_token
            else:
                piece = chr(token_id)
            if not (skip_special_tokens and token_id in self.all_special_ids):
                pieces.append(piece)
        return "".join(pieces)

    def convert_tokens_to_ids(self, token):
        if token == self.im_end_token:
            return self.im_end_token_id
        return self.unk_token_id

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return "KIMI:<think>"


class FakeBatch:
    def __init__(self, batch, non_tensor_batch, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = dict(meta_info or {})

    def __len__(self):
        return self.batch["old_log_probs"].shape[0]


class TestDPCAOPD(unittest.TestCase):
    def test_chatml_parser_preserves_reasoning_and_generation_mode(self):
        prompt = (
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nU<|im_end|>\n"
            "<|im_start|>assistant\n<think>r</think>answer<|im_end|>\n"
            "<|im_start|>user\nnext<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        messages, thinking = parse_qwen_chatml_generation_prompt(prompt)
        self.assertTrue(thinking)
        self.assertEqual(messages[2]["reasoning_content"], "r")
        self.assertEqual(messages[2]["content"], "answer")

    def test_dpca_finds_minimal_synchronized_chunks(self):
        student = PieceTokenizer({1: "hel", 2: "lo", 3: " world"})
        teacher = PieceTokenizer({10: "h", 11: "el", 12: "lo", 13: " world"})
        chunks = dpca_minimal_chunks(
            [1, 2, 3],
            [10, 11, 12, 13],
            student_tokenizer=student,
            teacher_tokenizer=teacher,
        )
        self.assertEqual(
            [
                (chunk.student_start, chunk.student_end, chunk.teacher_start, chunk.teacher_end)
                for chunk in chunks
            ],
            [(0, 1, 0, 2), (1, 2, 2, 3), (2, 3, 3, 4)],
        )

    def test_semantic_prior_credit_conserves_each_chunk(self):
        student = PieceTokenizer({1: "hel", 2: "lo"})
        teacher = PieceTokenizer({10: "h", 11: "el", 12: "lo"})
        chunks = dpca_minimal_chunks(
            [1, 2],
            [10, 11, 12],
            student_tokenizer=student,
            teacher_tokenizer=teacher,
        )
        advantages, targets, _, error, clipped = dpca_semantic_prior_credit(
            [-2.0, -3.0],
            [-1.0, -2.0, -2.0],
            chunks,
            log_prob_min_clamp=None,
            loss_max_clamp=None,
        )
        self.assertEqual(advantages, [-1.0, 1.0])
        self.assertEqual(targets, [-3.0, -2.0])
        self.assertLess(error, 1e-8)
        self.assertEqual(clipped, 0)

    def test_remote_score_and_attach_supervises_visible_and_first_stop_tokens(self):
        chatml = (
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nU<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        student = PieceTokenizer(
            {
                1: chatml,
                10: "hel",
                11: "lo",
                98: "<|endoftext|>",
                99: "<|im_end|>",
            },
            special_ids={98, 99},
        )
        teacher = CharacterTeacherTokenizer()
        settings = DPCAOPDSettings(
            enabled=True,
            teacher_base_url="http://teacher/v1",
            teacher_model="teacher",
            teacher_tokenizer_path="unused",
            strict_echo_token_check=True,
            alignment_dump_rows=1,
        )

        def fake_request(url, payload, headers, timeout):
            self.assertEqual(url, "http://teacher/v1/completions")
            self.assertEqual(headers["Content-Type"], "application/json")
            self.assertGreater(timeout, 0)
            choices = []
            for index, prompt in enumerate(payload["prompt"]):
                self.assertTrue(prompt.endswith(teacher.im_end_token))
                token_ids = teacher.encode(prompt, add_special_tokens=False)
                tokens = [
                    teacher.decode([token_id], skip_special_tokens=False)
                    for token_id in token_ids
                ]
                token_logprobs = [None] + [-1.0] * (len(tokens) - 1)
                token_logprobs[-1] = -0.25
                choices.append(
                    {
                        "index": index,
                        "text": "x",
                        "logprobs": {
                            "tokens": tokens + ["x"],
                            "token_logprobs": token_logprobs + [-1.0],
                        },
                    }
                )
            return {"choices": choices}

        prompt_ids = [1]
        response_ids = [10, 11, 99, 98]
        prompt_digest = prompt_token_digest(prompt_ids)
        response_digest = prompt_token_digest(response_ids)
        batch = FakeBatch(
            batch={
                "prompts": torch.tensor([[0, 1]]),
                "responses": torch.tensor([[10, 11, 99, 98, 0]]),
                AGENTMEMORY_GENERATION_PROMPT_LENGTH: torch.tensor([1]),
                AGENTMEMORY_PACKED_PROMPT_LENGTH: torch.tensor([1]),
                AGENTMEMORY_GENERATION_RESPONSE_LENGTH: torch.tensor([4]),
                AGENTMEMORY_PACKED_RESPONSE_LENGTH: torch.tensor([4]),
                "old_log_probs": torch.tensor([[-2.0, -3.0, -1.0, -0.5, 0.0]]),
                "response_mask": torch.tensor(
                    [[1, 1, 1, 1, 0]], dtype=torch.bool
                ),
                "ppo_valid_sample_mask": torch.tensor([True]),
            },
            non_tensor_batch={
                AGENTMEMORY_ACTION_TEXT: ["hello"],
                AGENTMEMORY_GENERATION_PROMPT_DIGEST: [prompt_digest],
                AGENTMEMORY_PACKED_PROMPT_DIGEST: [prompt_digest],
                AGENTMEMORY_GENERATION_RESPONSE_DIGEST: [response_digest],
                AGENTMEMORY_PACKED_RESPONSE_DIGEST: [response_digest],
                AGENTMEMORY_STEP_RECORD_JSON: [
                    {
                        "action": "hello",
                        "response_token_ids": response_ids,
                    }
                ],
            },
        )
        scorer = RemoteDPCAOPDScorer(
            settings,
            student_tokenizer=student,
            teacher_tokenizer=teacher,
            request_fn=fake_request,
        )
        scores = scorer.submit(batch).result(timeout=5)
        with tempfile.TemporaryDirectory() as dump_dir:
            settings = DPCAOPDSettings(
                **{
                    **settings.__dict__,
                    "alignment_dump_dir": dump_dir,
                }
            )
            metrics = attach_dpca_opd_advantages(
                batch,
                scores,
                settings,
                student_tokenizer=student,
                teacher_tokenizer=teacher,
                global_step=1,
            )
            dump = json.loads(Path(dump_dir, "step_000001.json").read_text())
            self.assertEqual(dump["schema"], "dpca_opd_alignment_v1")
        self.assertTrue(
            torch.equal(
                batch.batch[DPCA_OPD_TOKEN_MASK],
                torch.tensor([[True, True, True, False, False]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                batch.batch[DPCA_OPD_ADVANTAGES],
                torch.tensor([[-1.0, 1.0, 0.75, 0.0, 0.0]]),
            )
        )
        self.assertEqual(metrics["dpca_opd/aligned_student_tokens"], 3.0)
        self.assertEqual(metrics["dpca_opd/student_stop_tokens"], 2.0)
        self.assertEqual(metrics["dpca_opd/supervised_student_stop_tokens"], 1.0)
        self.assertEqual(metrics["dpca_opd/teacher_stop_tokens"], 1.0)
        self.assertEqual(metrics["dpca_opd/teacher_failure_rate"], 0.0)

    def test_dpca_reconstructs_canonically_equivalent_split_unicode(self):
        student = PieceTokenizer({1: "e", 2: "\u0301"})
        teacher = PieceTokenizer({10: "e", 11: "\u0301"})
        chunks = dpca_minimal_chunks(
            [1, 2],
            [10, 11],
            student_tokenizer=student,
            teacher_tokenizer=teacher,
        )
        self.assertEqual(len(chunks), 2)

    def test_runtime_digest_mismatch_fails_closed(self):
        chatml = (
            "<|im_start|>system\nS<|im_end|>\n"
            "<|im_start|>user\nU<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )
        student = PieceTokenizer(
            {1: chatml, 10: "hello", 99: "<|im_end|>"},
            special_ids={99},
        )
        teacher = CharacterTeacherTokenizer()
        batch = FakeBatch(
            batch={
                "prompts": torch.tensor([[1]]),
                "responses": torch.tensor([[10, 99]]),
                AGENTMEMORY_GENERATION_PROMPT_LENGTH: torch.tensor([1]),
                AGENTMEMORY_PACKED_PROMPT_LENGTH: torch.tensor([1]),
                AGENTMEMORY_GENERATION_RESPONSE_LENGTH: torch.tensor([2]),
                AGENTMEMORY_PACKED_RESPONSE_LENGTH: torch.tensor([2]),
                "old_log_probs": torch.tensor([[-2.0, -1.0]]),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                "ppo_valid_sample_mask": torch.tensor([True]),
            },
            non_tensor_batch={
                AGENTMEMORY_ACTION_TEXT: ["hello"],
                AGENTMEMORY_GENERATION_PROMPT_DIGEST: ["wrong"],
                AGENTMEMORY_PACKED_PROMPT_DIGEST: [prompt_token_digest([1])],
                AGENTMEMORY_GENERATION_RESPONSE_DIGEST: [
                    prompt_token_digest([10, 99])
                ],
                AGENTMEMORY_PACKED_RESPONSE_DIGEST: [
                    prompt_token_digest([10, 99])
                ],
                AGENTMEMORY_STEP_RECORD_JSON: [
                    {"action": "hello", "response_token_ids": [10, 99]}
                ],
            },
        )
        settings = DPCAOPDSettings(
            enabled=True,
            teacher_base_url="http://teacher/v1",
            teacher_model="teacher",
            teacher_tokenizer_path="unused",
        )
        scorer = RemoteDPCAOPDScorer(
            settings,
            student_tokenizer=student,
            teacher_tokenizer=teacher,
            request_fn=lambda *_: {},
        )
        with self.assertRaisesRegex(DPCAOPDError, "token digest differs"):
            scorer.submit(batch)

    def test_default_credit_has_no_nonpaper_clamps(self):
        settings = DPCAOPDSettings()
        self.assertIsNone(settings.log_prob_min_clamp)
        self.assertIsNone(settings.loss_max_clamp)

    def test_disabled_config_has_no_teacher_requirements(self):
        settings = DPCAOPDSettings.from_config({"enabled": False})
        self.assertFalse(settings.enabled)

    def test_alignment_rejects_different_surface_text(self):
        with self.assertRaises(DPCAOPDError):
            dpca_minimal_chunks(
                [1],
                [2],
                student_tokenizer=PieceTokenizer({1: "a"}),
                teacher_tokenizer=PieceTokenizer({2: "b"}),
            )


if __name__ == "__main__":
    unittest.main()
