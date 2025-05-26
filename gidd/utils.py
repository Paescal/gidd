import re
import math

import torch
import numpy as np
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



def get_position_metric(config, tokenizer):
    match config.sampling.position_metric:
        case "MDM_max":
            return partial(position_metric_MDM_max, tokenizer=tokenizer)
        case "MDM_margin":
            return partial(position_metric_MDM_margin, tokenizer=tokenizer)
        case "max":
            return position_metric_max
        case "margin":
            return position_metric_margin

def get_position_sampling_strategy(config):
    match config.sampling.position_sampling_strategy:
        case "all":
            return all_positions
        case "top_k_gumbel":
            return partial(sample_position_top_k_gumbel, k=config.sampling.position_sampling_strategy_args.top_k_gumbel.k, gumbel_noise_coefficient=config.sampling.position_sampling_strategy_args.top_k_gumbel.gumbel_noise_coefficient)
        case "top_k":
            return partial(sample_top_k, k=config.sampling.position_sampling_strategy_args.top_k, as_mask=True)
        case "top_p":
            return partial(sample_top_p, p=config.sampling.position_sampling_strategy_args.top_p, normalize_input=True, as_mask=True)
        case "min_p":
            return partial(sample_min_p, p=config.sampling.position_sampling_strategy_args.min_p, indices_only=True, as_mask=True)
        
def get_token_sampling_strategy(config, tokenizer):
    match config.sampling.token_sampling_strategy:
        case "MDM_categorical":
            return partial(sample_token_MDM_categorical, tokenizer=tokenizer)
        case "categorical":
            return sample_categorical
        case "top_k":
            return partial(sample_top_k, k=config.sampling.token_sampling_strategy_args.top_k)
        case "top_p":
            return partial(sample_top_p, p=config.sampling.token_sampling_strategy_args.top_p)
        case "min_p":
            return partial(sample_min_p, p=config.sampling.token_sampling_strategy_args.min_p)


@torch.no_grad()
def position_metric_MDM_max(z_t, probs, tokenizer):
    is_mask_token = (z_t == tokenizer.mask_token_id)
    probs = probs.clone()
    probs[..., tokenizer.mask_token_id] = 0
    metric = torch.max(probs, dim=-1).values
    return metric * is_mask_token

@torch.no_grad()
def position_metric_MDM_margin(z_t, probs, tokenizer):
    is_mask_token = (z_t == tokenizer.mask_token_id)
    probs = probs.clone()
    probs[..., tokenizer.mask_token_id] = 0
    top_2 = torch.topk(probs, 2, dim=-1).values
    metric = top_2[..., 0] - top_2[..., 1]
    return metric * is_mask_token

@torch.no_grad()
def position_metric_max(z_t, probs):
    metric = torch.max(probs, dim=-1).values
    return metric
    
    # Don't allow changing the token if the model thinks the current one is the best (this is flawed as is, because the mask token is always the best until almost the end)
    # max_indices = torch.max(probs, dim=-1).indices
    # z_t_is_max = (z_t == max_indices)

    # metric = torch.max(probs, dim=-1).values
    # metric = torch.where(z_t_is_max, 0, metric)
    # if diffusion_mask is not None:
    #     return metric * diffusion_mask
    # else:
    #     return metric

@torch.no_grad()
def position_metric_margin(z_t, probs):
    top_2 = torch.topk(probs, 2, dim=-1).values
    metric = top_2[..., 0] - top_2[..., 1]
    return metric
    
    # Don't allow changing the token if the model thinks the current one is the best (this is flawed as is, because the mask token is always the best until almost the end)
    # max_indices = torch.max(probs, dim=-1).indices
    # z_t_is_max = (z_t == max_indices)

    # top_2 = torch.topk(probs, 2, dim=-1).values
    # metric = top_2[..., 0] - top_2[..., 1]
    # metric = torch.where(z_t_is_max, 0, metric)
    # if diffusion_mask is not None:
    #     return metric * diffusion_mask
    # else:
    #     return metric

@torch.no_grad()
def all_positions(metric):
    return metric != 0

@torch.no_grad()
def sample_position_top_k_gumbel(metric, k, gumbel_noise_coefficient=0, generator=None):
    metric_is_non_zero_mask = metric != 0
    # TODO: what exactly is top k even supposed to do? Select k elements sampled categorically without replacement from tokens which are currently masks, but with probabilities only considering tokens for the values 1-9?
    # print(metric[0])
    # metric = metric / torch.sum(metric, dim=-1, keepdim=True)
    # print(f"metric shape: {metric.shape}")
    if gumbel_noise_coefficient > 0:
        metric = metric + torch.distributions.gumbel.Gumbel(0, gumbel_noise_coefficient).sample(metric.shape).to(metric.device, dtype=metric.dtype)
        metric = metric * metric_is_non_zero_mask
    top_k_thresholds = torch.topk(metric, k, dim=-1).values[..., -1].unsqueeze(-1)
    top_k_mask = metric >= top_k_thresholds
    return top_k_mask

@torch.no_grad()
def sample_categorical(probs, end_index=-1, generator=None):
    # return torch.distributions.Categorical(probs=probs).sample()
    uniform = torch.rand(probs.shape[:-1], dtype=probs.dtype, device=probs.device, generator=generator).unsqueeze(-1)
    cumprobs = probs.cumsum(-1)
    cumprobs[..., end_index:] = 1 + 1e-4
    samples = torch.searchsorted(cumprobs, uniform, right=True).squeeze(-1)
    return samples

@torch.no_grad()
def sample_token_MDM_categorical(probs, tokenizer, generator=None):
    probs = probs.clone()
    probs[..., tokenizer.mask_token_id] = 0
    probs = probs / probs.sum(-1, keepdim=True)
    return sample_categorical(probs, generator=generator)


@torch.no_grad()
# TODO: all_candidates=bool is only a temporary variable for testing selecting all candidates
def sample_top_k(metric, k, as_mask=False, all_candidates=False, generator=None):
    top_k_thresholds = torch.topk(metric, k + 1, dim=-1).values[..., -1].unsqueeze(-1)
    top_k_mask = metric > top_k_thresholds
    if all_candidates:
        if as_mask:
            return top_k_mask
    
    metric_masked = metric.masked_fill(~top_k_mask, 0)
    metric_masked_normalized = metric_masked / metric_masked.sum(-1, keepdim=True) # TODO: this assumes that metric is not all zeros
    chosen_indices = sample_categorical(metric_masked_normalized, generator=generator).unsqueeze(-1)
    if as_mask:
        return one_hot(chosen_indices.squeeze(-1), metric.shape[-1]).to(bool)
    else:
        return chosen_indices

    # candidates_metrics, candidates_indices = torch.topk(metric, k, dim=-1)
    # if all_candidates:
    #     if as_mask:
    #         candidates_mask = torch.zeros_like(metric, dtype=torch.bool, device=metric.device)
    #         candidates_mask.scatter_(-1, candidates_indices, True)
    #         return candidates_mask
    # candidates_probs = candidates_metrics / candidates_metrics.sum(-1, keepdim=True)

    # chosen_candidates = sample_categorical(candidates_probs, generator=generator).unsqueeze(-1)
    # chosen_indices = torch.gather(candidates_indices, -1, chosen_candidates).squeeze(-1)
    # if as_mask:
    #     return one_hot(chosen_indices, metric.shape[-1])
    # else:
    #     return chosen_indices

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

@torch.no_grad()
def correct_cells_score(samples_tokenized, solutions_tokenized):
    correct_cells = (samples_tokenized == solutions_tokenized).to(int)
    return torch.sum(correct_cells, dim=-1)

@torch.no_grad()
def row_col_box_set_score(sudoku):
    row_sets = [set(row) for row in sudoku]
    col_sets = [set(col) for col in sudoku.T]
    box_size = int(len(sudoku) ** 0.5)
    box_sets =[set(sudoku[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size].flatten()) for i in range(box_size) for j in range(box_size)]
    score = 0
    for curr_set in row_sets + col_sets + box_sets:
        for i in range(len(sudoku)):
            if str(i + 1) in curr_set:
                score += 1
    return score

@torch.no_grad()
def score_sudoku(samples_tokenized, diffusion_mask, solutions_tokenized, tokenizer):
    seq_len_original = samples_tokenized.shape[-1]
    sudoku_size = int(seq_len_original ** 0.5)
    seq_len = sudoku_size ** 2
    if seq_len_original == seq_len + 2:
        samples_tokenized = samples_tokenized[:, 1:-1]
        diffusion_mask = diffusion_mask[:, 1:-1]
        solutions_tokenized = solutions_tokenized[:, 1:-1]
    elif seq_len_original != seq_len:
        raise ValueError(f"Unexpected sequence length: {seq_len_original}, expected {seq_len + 2} or {seq_len}")
    
    num_samples = samples_tokenized.shape[0]
    cells_score = correct_cells_score(samples_tokenized, solutions_tokenized)
    given_cells = torch.sum((diffusion_mask == 0).to(int), dim=-1)
    filled_cells_score = cells_score - given_cells
    max_filled_cells_score = seq_len - given_cells
    # mean_filled_cells_score_fraction = torch.sum(filled_cells_score) / torch.sum(max_filled_cells_score)
    mean_filled_cells_score_fraction = torch.mean(filled_cells_score / max_filled_cells_score)

    fully_correct_samples_fraction = torch.mean((cells_score == (seq_len)).float())

    samples_decoded = np.array([tokenizer.decode(samples_tokenized[i], skip_special_tokens=False, clean_up_tokenization_spaces=False).split() for i in range(num_samples)])
    samples_decoded = samples_decoded.reshape((-1, sudoku_size, sudoku_size))
    set_score = [row_col_box_set_score(sample) for sample in samples_decoded]
    max_set_score = sudoku_size * sudoku_size * 3
    mean_set_score_fraction = np.mean(set_score) / max_set_score
    valid_sudoku_fraction = np.mean(np.array(set_score) == max_set_score)
    
    return {
        "correctly_filled_cells": mean_filled_cells_score_fraction,
        "correct_solution": fully_correct_samples_fraction,
        "set_score": mean_set_score_fraction,
        "valid_sudoku": valid_sudoku_fraction,
    }



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