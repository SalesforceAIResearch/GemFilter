import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)
from .snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)

# https://github.com/huggingface/transformers/blob/v4.43-release/src/transformers/models/mistral/modeling_mistral.py

def mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    # [SnapKV] register kv_cluster
    init_snapkv(self)
    
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # [SnapKV] update below
    kv_seq_len = key_states.shape[-2]
    if kv_seq_len == 1 and self.layer_idx == 0:
        position_ids += self.position_ids_margin
    else:
        self.position_ids_margin = 0
    # [SnapKV] update above
    
    if past_key_value is not None:
        kv_seq_len += cache_position[0]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    # repeat k/v heads if n_kv_heads < n_heads
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        # Activate slicing cache only if the config has a value `sliding_windows` attribute
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window

            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]

            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                    f" {past_key.shape}"
                )
            
            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        if q_len > 1:  # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            past_key_value.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            self.position_ids_margin = kv_seq_len - key_states_compress.shape[-2]
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        # Handle the case where the model is quantized
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    # print('layer id', self.layer_idx, 'query_states', query_states.shape, 'key_states', key_states.shape, 'value_states', value_states.shape, 'kv_seq_len', kv_seq_len, 'dropout_rate', dropout_rate, 'use_sliding_windows', use_sliding_windows)
    # [SnapKV] change attention_mask to None
    # print('layer id', self.layer_idx, 'query_states', query_states.shape, 'key_states', key_states.shape, 'value_states', value_states.shape, 'attention_mask', attention_mask.shape, 'kv_seq_len', kv_seq_len, 'dropout_rate', dropout_rate, 'use_sliding_windows', use_sliding_windows)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=getattr(self.config, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
