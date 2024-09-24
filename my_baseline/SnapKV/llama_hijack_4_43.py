import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)
from .snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)

# https://github.com/huggingface/transformers/blob/v4.43-release/src/transformers/models/llama/modeling_llama.py

def llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_snapkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
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
    
    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('key_states.shape:', key_states.shape)
        if q_len > 1:  # [SnapKV] add kv_cluster
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            self.position_ids_margin = kv_seq_len - key_states_compress.shape[-2]
        else:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
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

    attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            # position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
