import os
from dataclasses import dataclass
from typing import List, Literal
from transformers import PreTrainedTokenizer
import torch

# Rollout thinking is off by default so legacy runs keep emitting bare actions.
# Set AGENTMEMORY_ENABLE_THINKING=1 to let the model reason in a <think> block
# before acting -- this task needs it, since choosing the right product means
# reading candidate prices/ratings and comparing them against the Goal, which a
# bare action cannot express.
def _agentmemory_thinking_enabled() -> bool:
    return os.environ.get("AGENTMEMORY_ENABLE_THINKING", "0").strip().lower() in ("1", "true", "yes", "on")


def _agentmemory_reasoning_enabled() -> bool:
    """Allow explicit ReAct reasoning without enabling Qwen native thinking."""
    return os.environ.get("AGENTMEMORY_ALLOW_REASONING", "0").strip().lower() in ("1", "true", "yes", "on")


# The system prompt is built from three parts. The intro and the action-space
# contract are identical in both modes; only the reply-format rule differs, so
# that the rule never contradicts whether the chat template opened a <think>
# block (forbidding <think> while the template opens one would be self-defeating).
_AGENTMEMORY_INTRO = (
    "You are acting inside AgentMemoryGym, a native WebShop bundled-shopping environment. "
)

# No-thinking mode: the entire reply must be one bare action.
_AGENTMEMORY_REPLY_RULE_NO_THINKING = (
    "Reply with exactly one executable action and nothing else: either one native browser "
    "action or one uppercase memory-tool JSON action. Output excludes angle-bracket "
    "placeholders, markdown, explanations, Thought/Action labels, and <think> blocks. "
)

# Thinking mode: optional reasoning inside one <think> block, then one bare action.
_AGENTMEMORY_REPLY_RULE_THINKING = (
    "You may first reason inside a single <think>...</think> block. After the closing "
    "</think>, reply with exactly one executable action and nothing else: either one native "
    "browser action or one uppercase memory-tool JSON action. Apart from that optional "
    "<think> block, output excludes angle-bracket placeholders, markdown, explanations, and "
    "Thought/Action labels. "
)

# ReAct reasoning mode: the chat template keeps native thinking disabled, while
# the policy emits a short Thought plus the one Action executed by the env.
_AGENTMEMORY_REPLY_RULE_REASONING = (
    "Reply with exactly two labeled fields. Write `Thought:` followed by brief free-form "
    "reasoning, then write `Action:` followed by exactly one executable action: either one "
    "native browser action or one uppercase memory-tool JSON action. The environment executes "
    "only the action after the final `Action:` label, while PPO trains the complete sampled "
    "Thought-and-Action response. Output excludes markdown and <think> blocks. "
)

_AGENTMEMORY_ACTION_CONTRACT = (
    "Native browser actions use square-bracket syntax. search[keywords] runs a catalog "
    "search whose keywords are concrete product wording such as a visible product name or "
    "title; a bare category word or attribute alone matches little. click[value] clicks one "
    "currently displayed clickable value, exactly as shown in the available-actions list: an "
    "asin opens that product page, and the page also exposes navigation such as "
    "click[Back to Search], click[< Prev], click[Next >], click[Description], click[Features], "
    "click[Reviews], option values, and click[Buy Now]. A product page shows title, price, "
    "rating, sub-pages, and selectable options. click[Buy Now] on the open product commits the "
    "purchase of the current shopping session; a correct purchase advances to the next session "
    "and an incorrect purchase ends the episode with reward -0.01 and no retry. The visible "
    "available-actions list enumerates the clickable values valid on the current page. "
    "Memory tools use one uppercase name followed by one JSON object. ADD requires key:string "
    "and value:string and returns a new memory_id while storing exactly the text you wrote. "
    "UPDATE requires memory_id:string and value:string and replaces that memory value. DELETE "
    "requires memory_id:string and removes it. RETRIEVE requires query:string and top_k=3 and "
    "matches text you previously wrote to long-term memory with ADD (facts carried over from "
    "earlier sessions), not the current page or catalog, exposing matches as visible C# items. "
    "SUMMARY requires text:string and a non-empty source_ids:list[string] of visible S#/C# ids "
    "and replaces active context with that summary. FILTER requires exactly one non-empty "
    "keep_ids:list[string] or drop_ids:list[string], plus scope set to active, session, or all, "
    "and only changes visible S#/C# context. Current-session browser trace is shown as S# "
    "items and retrieved or summarized memory as C# items. Current-session trace clears when a "
    "purchase advances the session. Long-term memory persists across shopping sessions and "
    "remains hidden until RETRIEVE exposes it."
)

_AGENTMEMORY_MEMORY_LIFECYCLE = (
    "A successful purchase clears the current session's page and short-term trace. Once "
    "you have selected the product for the current session, use ADD before click[Buy Now] "
    "to save one concise memory containing that product's identity and any visible "
    "attributes needed for later compatibility decisions. At the start of every later "
    "shopping session, use RETRIEVE to expose the relevant prior-purchase memories before "
    "choosing a compatible product. The environment does not perform these memory actions "
    "for you, and it does not reject an otherwise correct purchase when ADD was skipped."
)

# Full prompts for each mode: same intro and action contract, different reply rule.
AGENTMEMORY_ACTION_SYSTEM_PROMPT = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_NO_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_THINKING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)
AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING = (
    _AGENTMEMORY_INTRO
    + _AGENTMEMORY_REPLY_RULE_REASONING
    + _AGENTMEMORY_ACTION_CONTRACT
    + " "
    + _AGENTMEMORY_MEMORY_LIFECYCLE
)


def agentmemory_action_system_prompt() -> str:
    # Pick the reply rule that matches the active thinking mode so the prompt
    # never contradicts what the chat template does with <think>.
    if _agentmemory_thinking_enabled():
        return AGENTMEMORY_ACTION_SYSTEM_PROMPT_THINKING
    if _agentmemory_reasoning_enabled():
        return AGENTMEMORY_ACTION_SYSTEM_PROMPT_REASONING
    return AGENTMEMORY_ACTION_SYSTEM_PROMPT

def _normalize_chat_template_token_ids(encoded) -> List[int]:
    """Normalize tokenizer.apply_chat_template(..., tokenize=True) output.

    Some tokenizer implementations (observed with Qwen3.5-4B backed by
    Qwen2Tokenizer) return a BatchEncoding/dict rather than a plain list of
    token ids. Iterating that object yields string keys like "input_ids",
    which later breaks vLLM prompt validation.
    """
    if isinstance(encoded, dict) or (hasattr(encoded, "__getitem__") and "input_ids" in encoded):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    # A batched tokenizer output may be [[...]] for a single conversation.
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError(f"Expected one chat-template sequence, got batch size {len(encoded)}")
        encoded = encoded[0]
    token_ids = list(encoded)
    bad = [(i, type(x).__name__, repr(x)[:80]) for i, x in enumerate(token_ids) if not isinstance(x, int)]
    if bad:
        raise TypeError(f"Chat template produced non-integer token ids: {bad[:5]}")
    return token_ids


def apply_chat_template(tokenizer: PreTrainedTokenizer, conversations: list[dict[str, str]]) -> List[int]:
    """Tokenize a conversation into generation-prompt token ids.

    enable_thinking follows AGENTMEMORY_ENABLE_THINKING: when off (default) the
    template closes the assistant turn with an empty <think></think> so the model
    emits a bare action; when on, the template leaves the <think> block open so
    the model can reason before acting. Qwen3.5 tokenizers that do not accept the
    enable_thinking kwarg fall back to the plain call (unchanged behaviour).
    """
    enable_thinking = _agentmemory_thinking_enabled()
    try:
        encoded = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=True)
    return _normalize_chat_template_token_ids(encoded)


# Backward-compatible alias: existing callers/imports of the old name keep working
# and now honour the AGENTMEMORY_ENABLE_THINKING flag through the shared function.
def apply_chat_template_no_thinking(tokenizer: PreTrainedTokenizer, conversations: list[dict[str, str]]) -> List[int]:
    return apply_chat_template(tokenizer, conversations)



def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    def to_dict(self):
        return {'role': self.role, 'content': self.content}
    def __repr__(self):
        return str(self.to_dict())
    def __str__(self):
        return self.__repr_

class RolloutHandler:
    def __init__(
        self,
        messages: List[Message],
        task_name: str,
        item_id: int,
        score: float,
        done: bool,
        input_ids: List[int],
        prompt_ids: List[int],
        response_ids: List[int],
        attention_mask: List[int],
        prompt_attention_mask: List[int],
        response_attention_mask: List[int],
        position_ids: List[int],
        prompt_position_ids: List[int],
        response_position_ids: List[int],
        loss_mask: List[int],
        prompt_loss_mask: List[int],
        response_loss_mask: List[int],
        max_response_len: int = 8192,
        max_model_len: int = 32768   
    ):
        self.messages = messages
        self.task_name = task_name
        self.item_id = item_id
        self.score = score
        self.done = done
        self.input_ids = input_ids
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.attention_mask = attention_mask
        self.prompt_attention_mask = prompt_attention_mask
        self.response_attention_mask = response_attention_mask
        self.position_ids = position_ids
        self.prompt_position_ids = prompt_position_ids
        self.response_position_ids = response_position_ids
        self.loss_mask = loss_mask
        self.prompt_loss_mask = prompt_loss_mask
        self.response_loss_mask = response_loss_mask
        self.max_response_len = max_response_len
        self.max_model_len = max_model_len  
        self.format_config: dict = {
            "qwen": {
                "assistat_prefix_msg": "\n<|im_start|>assistant\n",
                "assistat_suffix_msg": "<|im_end|>",
                "user_prefix_msg": "\n<|im_start|>user\n",
                "user_suffix_msg": "<|im_end|>",
            }
        }

    def get_generation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        conversations = [
            msg.to_dict() for msg in self.messages
        ]
        return apply_chat_template(tokenizer, conversations)

    def get_latest_observation_prompt(self, tokenizer: PreTrainedTokenizer) -> List[int]:
        assert self.messages, "RolloutHandler has no messages."
        latest_user_message = self.messages[-1]
        assert latest_user_message.role == "user", (
            f"Latest-observation rollout expects the last message to be a user "
            f"observation, got role={latest_user_message.role!r}."
        )
        # Preserve the no-raw-history policy while giving the model the neutral
        # action and state-transition contract every round.
        system_prompt = agentmemory_action_system_prompt()
        return apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                latest_user_message.to_dict(),
            ],
        )
    
    
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='assistant', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["assistat_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["assistat_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        response = tokenizer.encode(content, add_special_tokens=False)
        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = response
            _loss_mask = [1] * len(response)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + response
            _loss_mask = [0] * len(prefix_token_ids) + [1] * len(response)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )
        append_token_ids += suffix_token_ids
        _loss_mask += [1] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
        
    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        format: Literal["qwen"] = "qwen",
    ) -> None:
        msg = Message(role='user', content=content)
        self.messages.append(msg)
        assert format in self.format_config.keys(), f"format {format} not supported"
        prefix_msg = self.format_config[format]["user_prefix_msg"]
        prefix_token_ids = tokenizer.encode(prefix_msg, add_special_tokens=False)
        suffix_msg = self.format_config[format]["user_suffix_msg"]
        suffix_token_ids = tokenizer.encode(suffix_msg, add_special_tokens=False)
        content_token_ids = tokenizer.encode(content, add_special_tokens=False)

        if self.input_ids[-len(prefix_token_ids) :] == prefix_token_ids:
            append_token_ids = content_token_ids
            _loss_mask = [0] * len(content_token_ids)
        elif self.input_ids[-len(suffix_token_ids) :] == suffix_token_ids:
            append_token_ids = prefix_token_ids + content_token_ids
            _loss_mask = [0] * len(prefix_token_ids) + [0] * len(content_token_ids)
        else:
            max_len = max(len(prefix_token_ids), len(suffix_token_ids))
            raise ValueError(
                f"""Unsupported end of message format:
                {tokenizer.decode(self.input_ids[-max_len:])}, {tokenizer.decode(self.input_ids)=}"""
            )

        append_token_ids += suffix_token_ids
        _loss_mask += [0] * len(suffix_token_ids)
        self.input_ids += append_token_ids
        _attention_mask = [1] * len(append_token_ids)
        self.attention_mask += _attention_mask
        _delta_position_ids = [pos_id for pos_id in range(1, len(append_token_ids) + 1)]
        last_position_ids = self.position_ids[-1]
        _position_ids = [pos_id + last_position_ids for pos_id in _delta_position_ids]
        self.loss_mask += _loss_mask
        self.position_ids += _position_ids
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Rollout Handler has different length of {len(self.input_ids)=},
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""
        
    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]
