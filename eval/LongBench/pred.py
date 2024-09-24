import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from my_utils.my_generation import set_topk, my_greedy_generate_selection, my_greedy_generate_standard
from my_utils.load_model import load_model

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", 
        "llama-3.1-8b-instruct", "llama-3.1-8b-instruct-gemfilter", "llama-3.1-8b-instruct-snapkv", "llama-3.1-8b-instruct-h2o", 
        "mistral-nemo-instruct-2407", "mistral-nemo-instruct-2407-gemfilter", "mistral-nemo-instruct-2407-snapkv", "mistral-nemo-instruct-2407-h2o", 
        "phi-3.5-mini-instruct", "phi-3.5-mini-instruct-gemfilter", "phi-3.5-mini-instruct-snapkv", "phi-3.5-mini-instruct-h2o"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--topk', type=int, default=None, help='KV cache size')
    parser.add_argument('--select_layer_idx', type=int, default=None, help='use which layer as selection')
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, model_name, model2path, out_path, topk, select_layer_idx):
    model, tokenizer, modified = load_model_and_tokenizer(
        model2path[model_name], model_name)
    device = model.device
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        if modified:
            set_topk(model, topk, mode=modified)
        if 'gemfilter' in model_name:
            output = my_greedy_generate_selection(
                input['input_ids'], input['attention_mask'], model, tokenizer, max_gen_len=max_gen, select_layer_idx=select_layer_idx)
        else:
            output = my_greedy_generate_standard(input['input_ids'], input['attention_mask'], model, tokenizer, max_gen_len=max_gen)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = output
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    modified = None
    if "llama" in model_name or "mistral" in model_name or "phi" in model_name:
        flash_attention_2 = True
        if 'gemfilter' in model_name:
            modified = 'gemfilter'
            path = path.split("-gemfilter")[0]
        elif 'snapkv' in model_name:
            modified = 'snapkv'
            path = path.split("-snapkv")[0]
        elif 'h2o' in model_name:
            modified = 'h2o'
            path = path.split("-h2o")[0]
            flash_attention_2 = False
        model, tokenizer = load_model(
            path, modified=modified, torch_dtype=torch.float16, flash_attention_2=flash_attention_2)
    model = model.eval()
    return model, tokenizer, modified


def get_data_folder_name(args):
    if args.e:
        data = load_dataset('THUDM/LongBench',
                            f"{dataset}_e", split='test')
        folder_name = 'pred_e'
    else:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        folder_name = 'pred'

    if args.topk is None:
        topk = ''
    else:
        topk = f'-{args.topk}'
    
    if args.select_layer_idx is None or args.select_layer_idx < 0:
        select_layer_idx = ''
    else:
        select_layer_idx = f'-layer-{args.select_layer_idx}'
    
    if not os.path.exists(f"{folder_name}/{args.model}{select_layer_idx}{topk}"):
        os.makedirs(f"{folder_name}/{args.model}{select_layer_idx}{topk}")
    out_path = f"{folder_name}/{args.model}{select_layer_idx}{topk}/{dataset}.jsonl"
        
    return data, out_path

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = 1 # torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
                    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
                    "passage_count", "passage_retrieval_en"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print(dataset)
        data, out_path = get_data_folder_name(args)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                                                  max_gen, prompt_format, dataset, model_name, model2path, \
                                                  out_path, args.topk, args.select_layer_idx))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
