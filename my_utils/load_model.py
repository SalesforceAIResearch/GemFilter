# transformers.__version__ == '4.43.3'
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES, MistralForCausalLM
from transformers.models.phi3.modeling_phi3 import PHI3_ATTENTION_CLASSES, Phi3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

def my_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_last_logits_only = False # edit this line for memory save
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if output_last_logits_only:
            logits = self.lm_head(hidden_states[:,-1:,:])
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# LLAMA_ATTENTION_CLASSES = {
#     "eager": LlamaAttention,
#     "flash_attention_2": LlamaFlashAttention2,
#     "sdpa": LlamaSdpaAttention,
# # }
LlamaForCausalLM.forward = my_forward
MistralForCausalLM.forward = my_forward
Phi3ForCausalLM.forward = my_forward

def load_model(model_id, modified=None, torch_dtype=torch.float16, device_map='auto', flash_attention_2=False):
    if flash_attention_2:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'
    if modified == 'gemfilter':
        from my_baseline.GemFilter.llama_select_attention import LlamaSelectAttention
        from my_baseline.GemFilter.mistral_select_attention import MistralSelectAttention
        from my_baseline.GemFilter.phi3_select_attention import Phi3SelectAttention
        LLAMA_ATTENTION_CLASSES[attn_implementation] = LlamaSelectAttention
        MISTRAL_ATTENTION_CLASSES[attn_implementation] = MistralSelectAttention
        PHI3_ATTENTION_CLASSES[attn_implementation] = Phi3SelectAttention
    elif modified == 'snapkv':
        assert flash_attention_2 is True
        from my_baseline.SnapKV.monkeypatch import replace_llama, replace_mistral, replace_phi3
        replace_llama()
        replace_mistral()
        replace_phi3()
    elif modified == 'h2o':
        assert flash_attention_2 is False
        from my_baseline.H2O.llama_h2o_attention import LlamaH2OtAttention
        from my_baseline.H2O.mistral_h2o_attention import MistralH2OAttention
        from my_baseline.H2O.phi3_h2o_attention import Phi3H2OAttention
        LLAMA_ATTENTION_CLASSES[attn_implementation] = LlamaH2OtAttention
        MISTRAL_ATTENTION_CLASSES[attn_implementation] = MistralH2OAttention
        PHI3_ATTENTION_CLASSES[attn_implementation] = Phi3H2OAttention
    else:
        assert modified is None

    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            attn_implementation=attn_implementation, 
                                            torch_dtype=torch_dtype, 
                                            device_map=device_map
                                            ).eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
