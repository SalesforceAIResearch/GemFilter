# Minimal version of LongBench

This is a very minimal implementation (with slightly modification) of [LongBench](https://github.com/THUDM/LongBench).

You should be able to reproduce the results shown on our report.

Example:

First, run prediction:
```bash
python pred.py\
  --model llama-3.1-8b-instruct-gemfilter\
  --topk 1024\
  --select_layer_idx 13
```

Then, run evaluation:

```bash
python eval.py --model llama-3.1-8b-instruct-gemfilter-layer-13-1024
```



# Citation

```
@article{bai2023longbench,
  title={LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding},
  author={Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2308.14508},
  year={2023}
}
```
