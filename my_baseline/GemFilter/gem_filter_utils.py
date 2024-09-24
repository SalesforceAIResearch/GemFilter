import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv


def standard_dis_index(data, queries, k, norm=1, pool=False, kernel_size=5, sum_over_heads=False):
    inner_product = torch.matmul(queries, data.transpose(-1, -2))
    inner_product = inner_product[:, :, 0, :]
    if sum_over_heads:
        inner_product = torch.sum(inner_product, dim=1, keepdim=True)
    if pool:
        inner_product = F.avg_pool1d(
            inner_product, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    top_k = torch.topk(inner_product, k, dim=-1)
    indices = top_k.indices
    distances = top_k.values
    if norm != 1:
        distances = distances / norm
    return distances, indices


def find_context(self, query_states, key_states, print_idx_dis=False):
    b, h, n, d = key_states.shape
    if self.indecies is None and self.layer_idx == self.select_layer_idx:
        assert b == 1
        key_states_repeat = repeat_kv(
            key_states, self.num_key_value_groups)
        query_last_states = query_states[:, :, -1:, :]
        _, indices = standard_dis_index(key_states_repeat, query_last_states, min(
            self.topk, n), pool=True, sum_over_heads=True)
        self.indecies = indices
        if print_idx_dis:
            print(self.layer_idx, torch.min(torch.abs(indices-62383)))
    return
