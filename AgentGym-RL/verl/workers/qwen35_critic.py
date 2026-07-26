"""Token-level PPO critic support for Qwen3.5."""

from __future__ import annotations

from typing import Any

from torch import nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)


class Qwen3_5TokenValueModel(Qwen3_5PreTrainedModel):
    """Reuse Qwen3.5's native text backbone with one scalar per token."""

    base_model_prefix = "model"
    config_class = Qwen3_5TextConfig
    _no_split_modules = ["Qwen3_5DecoderLayer"]

    def __init__(self, causal_lm: Qwen3_5PreTrainedModel):
        config = causal_lm.config
        if getattr(config, "model_type", None) != "qwen3_5_text":
            raise ValueError(
                "Qwen3_5TokenValueModel requires a text-only Qwen3.5 model, "
                f"got {getattr(config, 'model_type', None)!r}."
            )
        if not isinstance(getattr(causal_lm, "model", None), Qwen3_5TextModel):
            raise TypeError("Qwen3.5 causal model does not expose Qwen3_5TextModel at .model.")

        config.num_labels = 1
        super().__init__(config)
        self.model = causal_lm.model
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self._init_weights(self.score)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=None,
        **kwargs: Any,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        logits = self.score(outputs.last_hidden_state)
        if return_dict is False:
            return (logits,) + outputs[1:]
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def load_qwen3_5_token_value_model(
    pretrained_model_name_or_path: str,
    *,
    config,
    torch_dtype,
    attn_implementation: str,
    trust_remote_code: bool,
):
    """Load the checkpoint's text model and replace its tied language head."""

    if getattr(config, "model_type", None) != "qwen3_5":
        raise ValueError("Qwen3.5 critic loading requires the top-level checkpoint config.")
    text_config = getattr(config, "text_config", None)
    if getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise ValueError("Qwen3.5 checkpoint config has no valid text_config.")

    causal_lm, loading_info = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        config=text_config,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        output_loading_info=True,
    )
    critic_module = Qwen3_5TokenValueModel(causal_lm)
    loading_info = dict(loading_info)
    missing_keys = list(loading_info.get("missing_keys", ()))
    if "score.weight" not in missing_keys:
        missing_keys.append("score.weight")
    loading_info["missing_keys"] = missing_keys
    return critic_module, loading_info
