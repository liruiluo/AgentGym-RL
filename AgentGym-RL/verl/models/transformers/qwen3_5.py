# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Packed-sequence and response-only fused PPO forwards for Qwen3.5.

This is a focused backport of verl commits 3ba8181 and 6a6242f.  AgentGym-RL
uses the text-only Qwen3.5 model for actor and critic training. The private
response-only dispatch combines AgentGym-RL's response-position selection with
VERL's fused linear cross-entropy kernels, without changing ordinary HF model
forwards.
"""

from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
)


QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
RESPONSE_FUSED_KERNEL_BACKENDS = frozenset({"torch", "triton"})
_ORIGINAL_CAUSAL_LM_FORWARD_ATTR = "_verl_qwen35_original_forward"


@dataclass
class Qwen3_5ResponseFusedPPOOutput:
    """Selected-token PPO outputs without a full-vocabulary logits tensor."""

    log_probs: torch.Tensor
    entropy: Optional[torch.Tensor]


def is_qwen3_5_model_type(model_type: str) -> bool:
    return model_type in QWEN35_MODEL_TYPES


def apply_qwen3_5_packed_forward_patch() -> bool:
    """Install packed-sequence forwards on Transformers' text layers.

    Returns ``True`` only when this call changed the process-global model
    classes. Repeated actor/critic initialization is intentionally idempotent.
    """

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5ForCausalLM,
        Qwen3_5GatedDeltaNet,
    )

    already_patched = (
        Qwen3_5DecoderLayer.forward is qwen3_5_decoder_layer_forward
        and Qwen3_5GatedDeltaNet.forward is qwen3_5_gated_delta_net_forward
        and Qwen3_5ForCausalLM.forward is qwen3_5_causal_lm_forward_dispatch
    )
    if not hasattr(Qwen3_5ForCausalLM, _ORIGINAL_CAUSAL_LM_FORWARD_ATTR):
        setattr(
            Qwen3_5ForCausalLM,
            _ORIGINAL_CAUSAL_LM_FORWARD_ATTR,
            Qwen3_5ForCausalLM.forward,
        )
    Qwen3_5DecoderLayer.forward = qwen3_5_decoder_layer_forward
    Qwen3_5GatedDeltaNet.forward = qwen3_5_gated_delta_net_forward
    Qwen3_5ForCausalLM.forward = qwen3_5_causal_lm_forward_dispatch
    return not already_patched


def qwen3_5_causal_lm_forward_dispatch(self, *args, **kwargs):
    """Route only explicitly marked PPO calls through the fused response head."""

    use_response_fused_ppo = bool(
        kwargs.pop("_verl_response_fused_ppo", False)
    )
    if use_response_fused_ppo:
        return qwen3_5_text_response_fused_ppo_forward(self, *args, **kwargs)
    original_forward = getattr(type(self), _ORIGINAL_CAUSAL_LM_FORWARD_ATTR)
    return original_forward(self, *args, **kwargs)


def qwen3_5_text_response_fused_ppo_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    response_positions: Optional[torch.LongTensor] = None,
    response_labels: Optional[torch.LongTensor] = None,
    response_fused_kernel_backend: str = "triton",
    calculate_entropy: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Qwen3_5ResponseFusedPPOOutput:
    """Project selected response states directly to PPO log-prob and entropy."""

    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "Qwen3.5 response-fused PPO requires one packed input row, got "
            f"{None if input_ids is None else tuple(input_ids.shape)}."
        )
    if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
        raise ValueError(
            "Qwen3.5 response-fused PPO requires a positive scalar temperature."
        )
    if response_positions is None or response_labels is None:
        raise ValueError(
            "Qwen3.5 response-fused PPO requires response positions and labels."
        )
    if response_positions.ndim != 1 or response_labels.ndim != 1:
        raise ValueError("Response positions and labels must both be rank-1.")
    if response_positions.shape != response_labels.shape:
        raise ValueError(
            "Response positions and labels must have identical shapes: "
            f"positions={tuple(response_positions.shape)} "
            f"labels={tuple(response_labels.shape)}."
        )
    if response_positions.numel() == 0:
        raise ValueError("Response-fused PPO received no selected tokens.")

    backend = str(response_fused_kernel_backend).lower()
    if backend not in RESPONSE_FUSED_KERNEL_BACKENDS:
        raise ValueError(
            "Unsupported response-fused PPO backend "
            f"{response_fused_kernel_backend!r}; expected one of "
            f"{sorted(RESPONSE_FUSED_KERNEL_BACKENDS)}."
        )
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    kwargs.pop("logits_to_keep", None)

    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise RuntimeError(
            "Qwen3.5 packed model returned an unexpected hidden-state shape: "
            f"{tuple(hidden_states.shape)}."
        )

    positions = response_positions.to(
        device=hidden_states.device,
        dtype=torch.long,
    ).contiguous()
    labels = response_labels.to(
        device=hidden_states.device,
        dtype=torch.long,
    ).contiguous()
    if torch.any(positions < 0) or torch.any(positions >= hidden_states.shape[1]):
        raise ValueError(
            "A response predecessor position falls outside the packed hidden states."
        )
    selected_hidden = hidden_states.squeeze(0).index_select(0, positions).contiguous()

    vocab_weights = self.lm_head.weight
    if hasattr(vocab_weights, "full_tensor"):
        vocab_weights = vocab_weights.full_tensor()
    if not vocab_weights.is_contiguous():
        raise RuntimeError("Qwen3.5 LM-head weights must be contiguous for fused PPO.")
    if torch.any(labels < 0) or torch.any(labels >= vocab_weights.shape[0]):
        raise ValueError("A response label falls outside the model vocabulary.")
    selected_hidden = selected_hidden.to(vocab_weights.dtype)

    if backend == "triton":
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        log_probs, entropy = linear_cross_entropy(
            selected_hidden,
            vocab_weights,
            labels,
            float(temperature),
            "none",
        )
        if not calculate_entropy:
            entropy = None
    else:
        from verl.utils.experimental.torch_functional import FusedLinearForPPO

        log_probs, entropy = FusedLinearForPPO()(
            hidden_states=selected_hidden,
            vocab_weights=vocab_weights,
            input_ids=labels,
            temperature=float(temperature),
            compute_entropy=calculate_entropy,
        )
    return Qwen3_5ResponseFusedPPOOutput(
        log_probs=log_probs,
        entropy=entropy,
    )


def _call_accepts_kwarg(fn, name: str) -> bool:
    params = signature(fn).parameters
    return name in params or any(
        parameter.kind == parameter.VAR_KEYWORD
        for parameter in params.values()
    )


def _prepare_packed_seq_idx(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor],
) -> torch.Tensor:
    try:
        from fla.ops.utils import prepare_sequence_ids

        return prepare_sequence_ids(
            cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        ).to(torch.int32).unsqueeze(0)
    except Exception:
        offsets = (
            cu_seqlens_cpu
            if cu_seqlens_cpu is not None
            else cu_seqlens.detach().cpu()
        ).tolist()
        seq_idx = [
            torch.full(
                (end - start,),
                index,
                device=cu_seqlens.device,
                dtype=torch.int32,
            )
            for index, (start, end) in enumerate(
                zip(offsets[:-1], offsets[1:], strict=True)
            )
        ]
        return torch.cat(seq_idx, dim=0).unsqueeze(0)


def _split_packed_args(
    cu_seqlens: torch.LongTensor,
    tensors: tuple[torch.Tensor, ...],
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    dim: int = 1,
):
    offsets = (
        cu_seqlens_cpu
        if cu_seqlens_cpu is not None
        else cu_seqlens.detach().cpu()
    ).tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        chunks = []
        for tensor in tensors:
            split_dim = dim if dim >= 0 else tensor.ndim + dim
            slices = [slice(None)] * tensor.ndim
            slices[split_dim] = slice(start, end)
            chunks.append(tensor[tuple(slices)])
        yield tuple(chunks)


class _ConvPrefixExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tails: torch.Tensor, group: dist.ProcessGroup):
        from fla.ops.cp.comm import conv_cp_send_recv_fwd

        ctx.group = group
        return conv_cp_send_recv_fwd(tails.contiguous(), group)

    @staticmethod
    def backward(ctx, grad_prefix: torch.Tensor):
        from fla.ops.cp.comm import conv_cp_send_recv_bwd

        return conv_cp_send_recv_bwd(grad_prefix.contiguous(), ctx.group), None


def _build_fla_cp_context(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor],
    conv_kernel_size: int,
):
    group = get_ulysses_sequence_parallel_group()
    if group is None:
        return None

    from fla.ops.cp.context import build_cp_context

    return build_cp_context(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=group,
        conv1d_kernel_size=conv_kernel_size,
    )


def _prepend_cp_conv_prefix(
    mixed_qkv: torch.Tensor,
    cp_context,
) -> tuple[torch.Tensor, int]:
    prefix_len = max(int(cp_context.conv1d_kernel_size or 0) - 1, 0)
    if prefix_len == 0:
        return mixed_qkv, 0

    if mixed_qkv.shape[-1] >= prefix_len:
        tails = mixed_qkv[..., -prefix_len:]
    else:
        tails = F.pad(mixed_qkv, (prefix_len - mixed_qkv.shape[-1], 0))

    prefix = _ConvPrefixExchange.apply(tails, cp_context.group)
    valid_prefix_len = min(
        prefix_len,
        int(cp_context.pre_num_conv_tokens or 0),
    )
    if valid_prefix_len < prefix_len:
        prefix = prefix.clone()
        prefix[..., : prefix_len - valid_prefix_len] = 0
    return torch.cat((prefix, mixed_qkv), dim=-1), prefix_len


def _as_channel_last_conv1d_input(x: torch.Tensor) -> torch.Tensor:
    if x.stride(1) == 1:
        return x
    return x.transpose(1, 2).contiguous().transpose(1, 2)


def _prepare_cp_conv_seq_idx(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor],
    prefix_len: int,
) -> torch.Tensor:
    seq_idx = _prepare_packed_seq_idx(cu_seqlens, cu_seqlens_cpu)
    if prefix_len == 0:
        return seq_idx
    prefix_idx = seq_idx[:, :1].expand(seq_idx.shape[0], prefix_len)
    return torch.cat((prefix_idx, seq_idx), dim=1)


def _packed_causal_conv1d_fallback(
    self,
    mixed_qkv: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    outputs = []
    for (segment,) in _split_packed_args(
        cu_seqlens,
        (mixed_qkv,),
        cu_seqlens_cpu=cu_seqlens_cpu,
        dim=2,
    ):
        outputs.append(
            F.silu(self.conv1d(segment)[:, :, : segment.shape[-1]])
        )
    return torch.cat(outputs, dim=-1)


def _packed_chunk_gated_delta_rule(
    self,
    query,
    key,
    value,
    g,
    beta,
    cu_seqlens,
    cu_seqlens_cpu,
    cp_context=None,
):
    kwargs = {
        "g": g,
        "beta": beta,
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
    }
    if cu_seqlens is None:
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    if cp_context is not None:
        if not _call_accepts_kwarg(
            self.chunk_gated_delta_rule,
            "cp_context",
        ):
            raise NotImplementedError(
                "Qwen3.5 Ulysses SP requires FLA "
                "chunk_gated_delta_rule cp_context support."
            )
        kwargs["cp_context"] = cp_context
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    if _call_accepts_kwarg(self.chunk_gated_delta_rule, "cu_seqlens"):
        kwargs["cu_seqlens"] = cu_seqlens
        if _call_accepts_kwarg(
            self.chunk_gated_delta_rule,
            "cu_seqlens_cpu",
        ):
            kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    outputs = []
    for q_i, k_i, v_i, g_i, beta_i in _split_packed_args(
        cu_seqlens,
        (query, key, value, g, beta),
        cu_seqlens_cpu=cu_seqlens_cpu,
    ):
        split_kwargs = dict(kwargs)
        split_kwargs["g"] = g_i
        split_kwargs["beta"] = beta_i
        out_i, _ = self.chunk_gated_delta_rule(
            q_i,
            k_i,
            v_i,
            **split_kwargs,
        )
        outputs.append(out_i)
    return torch.cat(outputs, dim=1), None


def _has_previous_state(cache_params, layer_idx: int) -> bool:
    marker = getattr(cache_params, "has_previous_state", False)
    return marker(layer_idx) if callable(marker) else bool(marker)


def qwen3_5_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
):
    if cu_seqlens is not None and cache_params is not None:
        raise NotImplementedError(
            "Packed Qwen3.5 linear attention does not support cached forward."
        )

    module = import_module(self.__class__.__module__)
    hidden_states = module.apply_mask_to_padding_states(
        hidden_states,
        attention_mask,
    )

    batch_size, seq_len, _ = hidden_states.shape
    cp_context = None
    model_cu_seqlens = cu_seqlens
    model_cu_seqlens_cpu = cu_seqlens_cpu
    if cu_seqlens is not None:
        if batch_size != 1:
            raise ValueError(
                "Packed Qwen3.5 linear attention expects batch size 1."
            )
        total_seq_len = int(cu_seqlens[-1].item())
        ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
        if total_seq_len != seq_len:
            if (
                ulysses_sp_size <= 1
                or total_seq_len % ulysses_sp_size != 0
                or seq_len != total_seq_len // ulysses_sp_size
            ):
                raise ValueError(
                    f"Packed cu_seqlens end {total_seq_len} does not "
                    f"match seq_len {seq_len}."
                )
            cp_context = _build_fla_cp_context(
                cu_seqlens,
                cu_seqlens_cpu,
                self.conv_kernel_size,
            )
            model_cu_seqlens = cp_context.cu_seqlens
            model_cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    use_precomputed_states = (
        cache_params is not None
        and _has_previous_state(cache_params, self.layer_idx)
        and seq_len == 1
    )
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(
        batch_size,
        seq_len,
        -1,
        self.head_v_dim,
    )
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        layer_cache = cache_params.layers[self.layer_idx]
        conv_state = layer_cache.conv_states
        recurrent_state = layer_cache.recurrent_states
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(
                mixed_qkv,
                (self.conv_kernel_size - mixed_qkv.shape[-1], 0),
            )
            if hasattr(cache_params, "update_conv_state"):
                cache_params.update_conv_state(conv_state, self.layer_idx)

        if self.causal_conv1d_fn is not None:
            conv_prefix_len = 0
            conv_input = mixed_qkv
            if cp_context is not None:
                conv_input, conv_prefix_len = _prepend_cp_conv_prefix(
                    mixed_qkv,
                    cp_context,
                )
            if model_cu_seqlens is not None:
                conv_input = _as_channel_last_conv1d_input(conv_input)
            seq_idx = (
                _prepare_cp_conv_seq_idx(
                    model_cu_seqlens,
                    model_cu_seqlens_cpu,
                    conv_prefix_len,
                )
                if model_cu_seqlens is not None
                else None
            )
            conv_output = self.causal_conv1d_fn(
                x=conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            mixed_qkv = (
                conv_output[..., conv_prefix_len:]
                if conv_prefix_len
                else conv_output
            )
        elif model_cu_seqlens is not None:
            if cp_context is not None:
                mixed_qkv, conv_prefix_len = _prepend_cp_conv_prefix(
                    mixed_qkv,
                    cp_context,
                )
                model_cu_seqlens = model_cu_seqlens + conv_prefix_len
                model_cu_seqlens[0] = 0
                if model_cu_seqlens_cpu is not None:
                    model_cu_seqlens_cpu = (
                        model_cu_seqlens_cpu + conv_prefix_len
                    )
                    model_cu_seqlens_cpu[0] = 0
                mixed_qkv = _packed_causal_conv1d_fallback(
                    self,
                    mixed_qkv,
                    model_cu_seqlens,
                    model_cu_seqlens_cpu,
                )[..., conv_prefix_len:]
            else:
                mixed_qkv = _packed_causal_conv1d_fallback(
                    self,
                    mixed_qkv,
                    model_cu_seqlens,
                    model_cu_seqlens_cpu,
                )
        else:
            mixed_qkv = F.silu(
                self.conv1d(mixed_qkv)[:, :, :seq_len]
            )

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(
        batch_size,
        seq_len,
        -1,
        self.head_k_dim,
    )
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(
        batch_size,
        seq_len,
        -1,
        self.head_v_dim,
    )

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = (
            _packed_chunk_gated_delta_rule(
                self,
                query,
                key,
                value,
                g,
                beta,
                model_cu_seqlens,
                model_cu_seqlens_cpu,
                cp_context,
            )
        )
    else:
        core_attn_out, last_recurrent_state = (
            self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        )

    if cache_params is not None:
        if hasattr(cache_params, "update_recurrent_state"):
            cache_params.update_recurrent_state(
                last_recurrent_state,
                self.layer_idx,
            )
        else:
            cache_params.recurrent_states[
                self.layer_idx
            ] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def qwen3_5_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    **kwargs,
):
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    cu_seqlens_cpu = kwargs.pop("cu_seqlens_cpu", None)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    return residual + hidden_states
