from __future__ import annotations

import unittest

from verl.utils.agentgym.rollout_context import (
    normalize_generation_record,
    validate_official_vllm_generation_record,
)


EOS_TOKEN_IDS = [151645, 151643]


def generation_record(
    token_ids: list[int],
    *,
    finish_reason: str,
    max_tokens: int,
    stop_reason: int | None = None,
    eos_token_ids: list[int] | None = None,
    pad_token_id: int | None = 151643,
) -> dict:
    eos_token_ids = EOS_TOKEN_IDS if eos_token_ids is None else eos_token_ids
    return normalize_generation_record(
        token_ids,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        max_tokens=max_tokens,
        backend_finish_reason=finish_reason,
        stop_reason=stop_reason,
        finish_reason_source="official_vllm",
        token_ids_are_exact=True,
    )


class OfficialVllmCompletionTests(unittest.TestCase):
    def test_terminal_eos_completion_remains_valid(self) -> None:
        record = generation_record(
            [10, 11, EOS_TOKEN_IDS[0]],
            finish_reason="stop",
            max_tokens=8,
        )

        self.assertIs(validate_official_vllm_generation_record(record), record)
        self.assertFalse(record["truncated"])

    def test_length_completion_keeps_exact_sampled_tokens(self) -> None:
        record = generation_record(
            [10, 11, 12, 13],
            finish_reason="length",
            max_tokens=4,
        )

        self.assertIs(validate_official_vllm_generation_record(record), record)
        self.assertEqual(record["token_ids"], [10, 11, 12, 13])
        self.assertTrue(record["truncated"])

    def test_length_completion_must_reach_the_generation_limit(self) -> None:
        record = generation_record(
            [10, 11, 12],
            finish_reason="length",
            max_tokens=4,
        )

        with self.assertRaisesRegex(RuntimeError, "max_response_tokens"):
            validate_official_vllm_generation_record(record)

    def test_length_completion_rejects_eos_or_stop_reason(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpectedly contains EOS"):
            validate_official_vllm_generation_record(
                generation_record(
                    [10, 11, 12, EOS_TOKEN_IDS[0]],
                    finish_reason="length",
                    max_tokens=4,
                )
            )

        with self.assertRaisesRegex(RuntimeError, "stop_reason=None"):
            validate_official_vllm_generation_record(
                generation_record(
                    [10, 11, 12, 13],
                    finish_reason="length",
                    max_tokens=4,
                    stop_reason=12,
                )
            )

    def test_accepts_non_qwen_tokenizer_metadata(self) -> None:
        record = generation_record(
            [10, 11, 2],
            finish_reason="stop",
            max_tokens=8,
            eos_token_ids=[2],
            pad_token_id=0,
        )

        self.assertIs(validate_official_vllm_generation_record(record), record)

    def test_rejects_duplicate_or_negative_tokenizer_metadata(self) -> None:
        duplicate = generation_record(
            [10, 2],
            finish_reason="stop",
            max_tokens=8,
            eos_token_ids=[2, 2],
            pad_token_id=0,
        )
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            validate_official_vllm_generation_record(duplicate)

        negative_pad = generation_record(
            [10, 2],
            finish_reason="stop",
            max_tokens=8,
            eos_token_ids=[2],
            pad_token_id=0,
        )
        negative_pad["tokenizer_pad_token_id"] = -1
        with self.assertRaisesRegex(RuntimeError, "pad token ID"):
            validate_official_vllm_generation_record(negative_pad)


if __name__ == "__main__":
    unittest.main()
