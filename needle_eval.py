import argparse
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.needle.utils import load_context, insert_needle
from my_utils.my_generation import set_topk, my_greedy_generate_selection, my_greedy_generate_standard
from my_utils.load_model import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, 
                    choices=['meta-llama/Meta-Llama-3.1-8B-Instruct', 
                             'mistralai/Mistral-Nemo-Instruct-2407',
                             'microsoft/Phi-3.5-mini-instruct']) # huggingface model id
parser.add_argument('--modified', type=str, default=None, choices=['gemfilter', 'snapkv', 'h2o']) # None for standard attention
parser.add_argument('--topk', type=int, default=1024, help='KV cache size')
parser.add_argument('--ctx_len', type=int, default=32000, help='haystack context token length')
args = parser.parse_args()

model_id = args.model
modified = args.modified 
topk = args.topk
ctx_len = args.ctx_len  

if args.modified == 'h2o':
    flash_attention_2 = False
else:
    flash_attention_2 = True

if model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
    select_layer_idx = 13  # 13, 14 out of 32
elif model_id == 'mistralai/Mistral-Nemo-Instruct-2407':
    select_layer_idx = 19  # 19 out of 40
elif model_id == 'microsoft/Phi-3.5-mini-instruct':
    select_layer_idx = 19  # 19 out of 32
else:
    raise NotImplementedError

torch_dtype=torch.float16
model, tokenizer = load_model(model_id, modified=modified, torch_dtype=torch_dtype, flash_attention_2=flash_attention_2)
if modified:
    set_topk(model, topk, mode=modified)

# Construct the Needle-in-a-HayStack Prompt
needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"

depth = 0.5
context = load_context(fpath="eval/needle/PaulGrahamEssays/*.txt", ctx_len=ctx_len)
context = insert_needle(context, needle, depth=depth)
needle_idx = context.find("The best thing to do in San Francisco is")
print("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
print(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 

prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
question = "What is the best thing to do in San Francisco?"
prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
# print(prompt) # feel the length of 100K

# Check how the model performs
prompt = tokenizer(prompt, return_tensors="pt")
input_ids = prompt['input_ids'].to(model.device)
attn_mask = prompt["attention_mask"].to(model.device)

print("After tokenization, there is %d tokens" % len(input_ids[0]))

with torch.no_grad():
    if modified == 'gemfilter':
        response = my_greedy_generate_selection(
            input_ids, attn_mask, model, tokenizer, max_gen_len=50, select_layer_idx=select_layer_idx, print_context=False)
    else:
        response = my_greedy_generate_standard(input_ids, attn_mask, model, tokenizer, max_gen_len=50)
print("Response:", response.split("\n")[0])
