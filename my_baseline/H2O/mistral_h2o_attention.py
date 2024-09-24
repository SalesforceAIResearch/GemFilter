# transformers.__version__ == '4.43.3'
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.mistral.modeling_mistral import MistralAttention, apply_rotary_pos_emb, repeat_kv

from .utils_h2o import H2OKVCache_LayerWise


class MistralH2OAttention(MistralAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=4096-32,
            recent_size=32,
            k_seq_dim=2,
            v_seq_dim=2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
        
        # [H2O] update below
        kv_seq_len = key_states.shape[-2]
        if kv_seq_len == 1 and self.layer_idx == 0:
            position_ids += self.position_ids_margin
        else:
            self.position_ids_margin = 0
        # [H2O] update above

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)
        
        # [H2O] move to ahead
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # [H2O] update below
        if past_key_value is not None and q_len > 1:
            self.kv_cache._clean_scores()
            past_key_value = self.kv_cache(key_states, query_states, self.layer_idx,
                                           1, past_key_value, attn_weights.detach().clone())
            self.position_ids_margin = kv_seq_len - self.kv_cache.cache_size
        # [H2O] update above
        
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
