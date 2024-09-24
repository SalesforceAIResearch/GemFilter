import torch

# reverse opertaion of repeat_kv
def sum_group(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(
        batch, num_heads // n_rep, n_rep, slen, head_dim)
    return hidden_states.sum(2)

class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_max_size = hh_size + recent_size
        self.cache_size = 0
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, key_states, query_states, layer_idx, num_key_value_groups, past_key_values, attn_score_cache):

        self._update_hh_score(attn_score_cache, num_key_value_groups)
        # check if prefix phase
        if past_key_values is None:
            return None
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, seq_len, head_dim = key_states.shape
        if seq_len < self.cache_max_size:
            self.cache_size = seq_len
            return past_key_values
        
        self.cache_size = self.cache_max_size
        # hh-selection
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len,
                                   device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(
            query_states.device)
        mask = mask.scatter(-1, keep_idx, 1)

        past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx].squeeze(
        )[mask].view(bsz, num_heads//num_key_value_groups, -1, head_dim)
        past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx].squeeze(
        )[mask].view(bsz, num_heads//num_key_value_groups, -1, head_dim)

        self.hh_score = self.hh_score[mask].view(
            num_heads//num_key_value_groups, self.cache_size)

        return past_key_values

    def _update_hh_score(self, attn_score_cache, num_key_value_groups):

        num_new_tokens = attn_score_cache.shape[2]
        attn_score_cache = sum_group(attn_score_cache, num_key_value_groups)
        attn_score_cache = attn_score_cache.sum(0).sum(1)

        if self.hh_score is None:
            self.hh_score = attn_score_cache
        else:
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    def _clean_scores(self):
        self.hh_score = None
        self.cache_size = 0
