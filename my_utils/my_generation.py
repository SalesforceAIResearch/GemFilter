import warnings

warnings.filterwarnings("ignore")

import torch

def get_layer_context(model, tokenizer, input_ids, layer_idx, print_context=False):
    decoder_layer = model.model.layers[layer_idx]
    idx = decoder_layer.self_attn.indecies[0, 0, :]
    values, _ = torch.sort(idx)
    values = values.to('cuda:0')
    new_input_ids = input_ids.gather(0, values)
    if print_context:
        print(tokenizer.decode(new_input_ids))
    return new_input_ids.unsqueeze(0)

def reduce_layer(model, layer_idx):
    original_layers = model.model.layers
    model.model.layers = model.model.layers[:layer_idx+1]
    return model, original_layers

def recover_layer(model, layers):
    model.model.layers = layers
    return model

def set_topk(model, topk, mode='gemfilter'):
    decoder_layers = model.model.layers
    for i in range(len(decoder_layers)):
        if mode == 'gemfilter':
            decoder_layers[i].self_attn.topk = topk
        elif mode == 'snapkv':
            decoder_layers[i].self_attn.config.max_capacity_prompt = topk 
        elif mode == 'h2o':
            recent_size = decoder_layers[i].self_attn.kv_cache.recent_size
            decoder_layers[i].self_attn.kv_cache.hh_size = topk - recent_size
            decoder_layers[i].self_attn.kv_cache.cache_max_size = topk
        else:
            raise NotImplementedError
    return

def set_select_mode(model, mode):
    decoder_layers = model.model.layers
    for i in range(len(decoder_layers)):
        decoder_layers[i].self_attn.select_mode = mode
    return

def set_select_layer(model, select_layer_idx):
    if select_layer_idx is None:
        select_layer_idx = model.model.layers[0].self_attn.select_layer_idx
    else:
        decoder_layers = model.model.layers
        for i in range(len(decoder_layers)):
            decoder_layers[i].self_attn.select_layer_idx = select_layer_idx
    return select_layer_idx

@torch.no_grad()
def my_greedy_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len):
    generated_ids = [pred_token_idx.item()]
    for _ in range(max_gen_len):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        if pred_token_idx == tokenizer.eos_token_id:
            break
        generated_ids.append(pred_token_idx.item())
    return generated_ids

@torch.no_grad()
def my_greedy_generate_selection(input_ids, attn_mask, model, tokenizer, max_gen_len=50, select_layer_idx=None, print_context=False):
    set_select_mode(model, True)
    select_layer_idx = set_select_layer(model, select_layer_idx)
    model, original_layers = reduce_layer(model, select_layer_idx)
    _ = model(input_ids, attention_mask=attn_mask, output_last_logits_only=True)
    
    new_input_ids = get_layer_context(
        model, tokenizer, input_ids[0], select_layer_idx, print_context=print_context)
    model = recover_layer(model, original_layers)
    
    set_select_mode(model, False)
    outputs = model(new_input_ids, attention_mask=attn_mask, output_last_logits_only=True)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    output_ids = my_greedy_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen_len)
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return response

@torch.no_grad()
def my_greedy_generate_standard(input_ids, attn_mask, model, tokenizer, max_gen_len=50):
    outputs = model(input_ids, attention_mask=attn_mask, output_last_logits_only=True)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    output_ids = my_greedy_generate(model, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen_len)
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return response
