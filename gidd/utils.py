import re
import math

import torch
from functools import partial
from torch.nn.functional import one_hot

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



def get_position_metric(config):
    match config.model.position_metric:
        case "max":
            return position_metric_max
        case "margin":
            return position_metric_margin

def get_position_sampling_strategy(config):
    match config.model.sampling_strategy:
        case "all":
            return all_positions
        case "top_k":
            return partial(sample_top_k, k=config.model.position_selection_startegy_args.top_k, as_mask=True)
        case "top_p":
            return partial(sample_top_p, p=config.model.position_selection_strategy_args.top_p, normalize_input=True, as_mask=True)
        case "min_p":
            return partial(sample_min_p, p=config.model.position_selection_strategy_args.min_p, indices_only=True, as_mask=True)
        
def get_token_sampling_strategy(config):
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
def position_metric_max(probs, diffusion_mask=None):
    if diffusion_mask is not None:
        return torch.max(probs, dim=-1).values * diffusion_mask
    else:
        return torch.max(probs, dim=-1).values

@torch.no_grad()
def position_metric_margin(probs, diffusion_mask=None):
    top_2 = torch.topk(probs, 2, dim=-1).values
    if diffusion_mask is not None:
        return (top_2[..., 0] - top_2[..., 1]) * diffusion_mask
    else:
        return top_2[..., 0] - top_2[..., 1]

@torch.no_grad()
def all_positions(metric):
    return metric != 0


@torch.no_grad()
def sample_categorical(probs, generator=None):
    # return torch.distributions.Categorical(probs=probs).sample()
    uniform = torch.rand(probs.shape[:-1], dtype=probs.dtype, device=probs.device, generator=generator).unsqueeze(-1)
    cumprobs = probs.cumsum(-1)
    cumprobs[..., -1] = 1 + 1e-4
    samples = torch.searchsorted(cumprobs, uniform, right=True).squeeze(-1)
    return samples

@torch.no_grad()
def sample_top_k(metric, k, as_mask=False, generator=None):
    candidates_metrics, candidates_indices = torch.topk(metric, k, dim=-1)
    candidates_probs = candidates_metrics / candidates_metrics.sum(-1, keepdim=True)

    chosen_candidates = sample_categorical(candidates_probs, generator=generator).unsqueeze(-1)
    chosen_indices = torch.gather(candidates_indices, -1, chosen_candidates).squeeze(-1)
    if as_mask:
        return one_hot(chosen_indices, metric.shape[-1])
    else:
        return chosen_indices

@torch.no_grad()
def sample_top_p(metric, p, as_mask=False, normalize_input=False, generator=None):
    if normalize_input:
        metric = metric / metric.sum(-1, keepdim=True)

    sorted_metric, sorted_indices = torch.sort(metric, dim=-1, descending=True)
    cumulative_metrics = sorted_metric.cumsum(-1)
    cumulative_metrics[..., -1] = 1 + 1e-4
    sorted_indices_to_ignore = cumulative_metrics >= p
    sorted_indices_to_ignore[..., 1:] = sorted_indices_to_ignore.clone()[..., :-1]
    sorted_indices_to_ignore[..., 0] = False
    masked_sorted_metrics = sorted_metric.masked_fill_(sorted_indices_to_ignore, 0)
    masked_sorted_metrics_normalize = masked_sorted_metrics / masked_sorted_metrics.sum(-1, keepdim=True)

    chosen_sorted_indices = sample_categorical(masked_sorted_metrics_normalize, generator=generator).unsqueeze(-1)
    chosen_indices = torch.gather(sorted_indices, -1, chosen_sorted_indices).squeeze(-1)
    if as_mask:
        return one_hot(chosen_indices, metric.shape[-1])
    else:
        return chosen_indices

@torch.no_grad()
def sample_min_p(metric, p, as_mask=False, generator=None):
    max_metric = torch.max(metric, dim=-1, keepdim=True).values
    metric_threshold = max_metric.expand_as(metric) * p
    metrics_to_ignore = metric < metric_threshold
    masked_metric = metric.clone().masked_fill_(metrics_to_ignore, 0)
    masked_metric_normalized = masked_metric / masked_metric.sum(-1, keepdim=True)

    chosen_indices = sample_categorical(masked_metric_normalized, generator=generator)
    if as_mask:
        return one_hot(chosen_indices, metric.shape[-1])
    else:
        return chosen_indices


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