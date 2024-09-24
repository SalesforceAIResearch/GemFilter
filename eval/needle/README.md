# Minimal version of Needle-in-a-Haystack

This is a very minimal implementation (with slightly modification) of [Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack).

You should be able to reproduce the results shown on our report.

Example:

First, run prediction:

```bash
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct\
    --modified gemfilter\
    --topk 1024\
    --select_layer_idx 13\
    --model_name_suffix gemfilter-layer-13-1024
```

Then, run visualization:

```bash
python visualize.py
```
