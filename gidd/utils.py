import re
import math

import torch
from functools import partial

def parse_dtype(dtype):
    if dtype == "fp16":
        return torch.float16
    elif dtype == "fp32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_lr(config, lr, step):
    lr_schedule = config.training.lr_schedule
    warmup_steps = config.training.warmup_steps
    num_train_steps = config.training.num_train_steps

    if lr_schedule == "constant":
        return lr
    elif lr_schedule == "linear":
        return lr * min(1, step / warmup_steps, 1 - (step - warmup_steps) / (num_train_steps - warmup_steps))
    elif lr_schedule == "cosine":
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            return lr * (0.1 + 0.9*(1 + math.cos(math.pi * (step - warmup_steps) / (num_train_steps - warmup_steps))) / 2)
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")



def get_position_selection_strategy(config):
    match config.model.sampling_strategy:
        case "all":
            return all_positions
        case "top_k":
            return partial(position_top_k, k=config.model.position_selection_startegy_args.top_k, probability_margin=config.model.position_selection_strategy_args.probability_margin)
        case "top_p":
            return partial(position_top_p, p=config.model.position_selection_strategy_args.top_p, probability_margin=config.model.position_selection_strategy_args.probability_margin)
        case "min_p":
            return partial(position_min_p, p=config.model.position_selection_strategy_args.min_p, probability_margin=config.model.position_selection_strategy_args.probability_margin)
        
def get_sampling_strategy(config):
    match config.model.sampling_strategy:
        case "categorical":
            return sample_categorical
        case "top_k":
            return partial(sample_top_k, k=config.model.sampling_startegy_args.top_k)
        case "top_p":
            return partial(sample_top_p, p=config.model.sampling_strategy_args.top_p)
        case "min_p":
            return partial(sample_min_p, p=config.model.sampling_strategy_args.min_p)

@torch.no_grad()
def all_positions(probs, generator=None):
    # return a mask of all 1s
    pass
@torch.no_grad()
def position_top_k(probs, k, probability_margin, generator=None):
    # compute metric for each position
    # sample position
    pass
@torch.no_grad()
def position_top_p(probs, p, probability_margin, generator=None):
    pass
@torch.no_grad()
def position_min_p(probs, p, probability_margin, generator=None):
    pass

@torch.no_grad()
def sample_categorical(probs, generator=None):
    # return torch.distributions.Categorical(probs=probs).sample()
    uniform = torch.rand(probs.shape[:-1], dtype=probs.dtype, device=probs.device, generator=generator).unsqueeze(-1)
    cumprobs = probs.cumsum(-1)
    cumprobs[..., -1] = 1 + 1e-4
    samples = torch.searchsorted(cumprobs, uniform, right=True).squeeze(-1)
    return samples

@torch.no_grad()
def sample_top_k(probs, k, generator=None):
    # sample updated token
    pass
@torch.no_grad()
def sample_top_p(probs, p, generator=None):
    pass
@torch.no_grad()
def sample_min_p(probs, p, generator=None):
    pass


def calculate_flops_per_batch(config, model, vocab_size, non_emb_params=None, method="hoffmann"):
    if method == "kaplan":
        assert non_emb_params is not None
        flops_per_token = 2 * (non_emb_params + config.model.n_blocks * config.model.hidden_size * config.model.max_seq_len)
        flops_per_sample = 3 * config.model.max_seq_len * flops_per_token
    elif method == "hoffmann":
        seq_len = config.model.max_seq_len
        d_model = config.model.hidden_size
        num_heads = config.model.n_heads
        mlp_ratio = 4
        num_layers = config.model.n_blocks

        emb_flops = 2 * seq_len * vocab_size * d_model
        attn_flops = (
            2 * 3 * seq_len * d_model**2
            + 2 * seq_len**2 * d_model
            + 3 * num_heads * seq_len**2
            + 2 * seq_len**2 * d_model
            + 2 * seq_len * d_model**2
        )
        mlp_flops = 2 * seq_len * 2 * d_model * (mlp_ratio * d_model)
        layer_flops = attn_flops + mlp_flops
        final_flops = 2 * seq_len * d_model * vocab_size

        if config.model.type == "diffusion":
            freq_dim = model.sigma_map.mlp[0].in_features
            cond_dim = config.model.cond_dim
            emb_flops += 2 * (freq_dim * d_model + d_model * d_model)
            layer_flops += 2 * (cond_dim * 6 * d_model)
            final_flops += 2 * (cond_dim * 2 * d_model)

        flops_per_sample = 3 * (emb_flops + num_layers * layer_flops + final_flops)
    else:
        raise ValueError(f"Unknown method: {method}")
    flops_per_batch = flops_per_sample * config.training.train_batch_size
    return flops_per_batch