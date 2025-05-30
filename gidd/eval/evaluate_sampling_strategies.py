import hydra
import tqdm
import torch
import numpy as np
import os

from functools import partial
from pathlib import Path
from gidd.utils import parse_dtype, score_sudoku
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler
from datasets import load_from_disk
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def sampling_config_to_str(sampling_config):
    position_metric = sampling_config.sampling.position_metric
    position_sampling_strategy = sampling_config.sampling.position_sampling_strategy
    k = sampling_config.sampling.position_sampling_strategy_args.top_k_gumbel.k
    noise_coefficient = sampling_config.sampling.position_sampling_strategy_args.top_k_gumbel.gumbel_noise_coefficient
    token_sampling_strategy = sampling_config.sampling.token_sampling_strategy
    return f"position_metric={position_metric}, position_sampling_strategy={position_sampling_strategy}, k={k}, noise_coefficient={noise_coefficient}, token_sampling_strategy={token_sampling_strategy}"

@hydra.main(config_path="../configs", config_name="evaluate_sampling_strategies", version_base="1.1")
def main(config):
# args pass batch size, ckpt_path, num_denoising_steps and min_p

    num_samples = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_paths = config.checkpoint_path
    models = []
    noise_schedules = []
    tokenizer = None
    ckpt_configs = []
    dtype = None

    for ckpt_path in ckpt_paths:
        ckpt_path = hydra.utils.to_absolute_path(ckpt_path)

        model, noise_schedule, ckpt_tokenizer, ckpt_config = load_checkpoint(ckpt_path, device=device)
        model.eval()
        ckpt_config.training.eval_batch_size = config.batch_size
        if tokenizer is None:
            tokenizer = ckpt_tokenizer
        if dtype is None:
            dtype = parse_dtype(ckpt_config.training.dtype)
        models.append(model)
        noise_schedules.append(noise_schedule)
        ckpt_configs.append(ckpt_config)

    # ds_eval = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/{ckpt_config.data.dataset_name}{('_' + ckpt_config.data.dataset_subset) if ckpt_config.data.dataset_subset else ''}/evaluate")
    ds_eval_easy = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/easy/test")
    ds_eval_hard = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/hard/test")
    datasets = {"easy": ds_eval_easy, "hard": ds_eval_hard}

    ds_names = list(datasets.keys())
    all_puzzles_tokenized = []
    all_solutions_tokenized = []
    all_diffusion_mask = []
    for ds_eval in datasets.values():
        ds_eval = ds_eval.select(range(num_samples))
        
        # print(f"Evaluating model from {ckpt_path} on {num_samples} samples")
        
        puzzles = ds_eval.select_columns(['puzzle'])
        # solutions = ds_eval.select_columns(['solution'])
        solutions = ds_eval.select_columns(['text'])

        puzzles_tokenized = torch.tensor(tokenizer(puzzles['puzzle'])['input_ids'])
        solutions_tokenized = torch.tensor(tokenizer(solutions['text'])['input_ids'])
        diffusion_mask = (puzzles_tokenized == tokenizer.mask_token_id).to(int)

        all_puzzles_tokenized.append(puzzles_tokenized)
        all_solutions_tokenized.append(solutions_tokenized)
        all_diffusion_mask.append(diffusion_mask)

    position_metrics = config.sampling.position_metric
    position_sampling_strategies = config.sampling.position_sampling_strategy
    top_k_gumbel_k = config.sampling.position_sampling_strategy_args.top_k_gumbel.k
    top_k_gumbel_noise_coefficient = config.sampling.position_sampling_strategy_args.top_k_gumbel.gumbel_noise_coefficient
    token_sampling_strategies = config.sampling.token_sampling_strategy

    for model, noise_schedule, ckpt_config, ckpt_path in zip(models, noise_schedules, ckpt_configs, ckpt_paths):
        print(f"\n\n{'-'*10}Evaluating model from {ckpt_path} on {num_samples} samples{'-'*10}\n")
        for position_sampling_strategy in position_sampling_strategies:
            for position_metric in position_metrics:
                for k in top_k_gumbel_k:
                    for noise_coefficient in top_k_gumbel_noise_coefficient:
                        for token_sampling_strategy in token_sampling_strategies:
                            sampling_config = {
                                "sampling": {
                                    "position_metric": position_metric,
                                    "position_sampling_strategy": position_sampling_strategy,
                                    "position_sampling_strategy_args": {
                                        "top_k_gumbel": {
                                            "k": k,
                                            "gumbel_noise_coefficient": noise_coefficient
                                        }
                                    },
                                    "token_sampling_strategy": token_sampling_strategy
                                }
                            }
                            sampling_config = dict_to_namespace(sampling_config)

                            sampler = get_sampler(ckpt_config, model, tokenizer, noise_schedule, sampling_config=sampling_config, compile_step=config.compilation.compile_torch, min_p=config.min_p)
                            model.eval()

                            all_samples = []
                            for i in range(len(datasets)):
                                samples = []
                                with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
                                    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                                        for j in range(0, num_samples, config.batch_size):
                                            bs = min(config.batch_size, num_samples - j)
                                            if ckpt_config.training.use_diffusion_mask:
                                                z_t = sampler.generate_from_given(all_puzzles_tokenized[i][j:j+bs], all_diffusion_mask[i][j:j+bs], config.num_denoising_steps, decode=False, show_progress=False, keep_history=False)
                                            else:
                                                z_t = sampler.generate_from_expected_t(all_puzzles_tokenized[i][j:j+bs], all_diffusion_mask[i][j:j+bs], config.num_denoising_steps, add_random_tokens=(ckpt_config.model.p_uniform == 0), decode=False, show_progress=False, keep_history=False)
                                            samples.append(z_t)
                                            pbar.update(bs)
                                samples = torch.cat(samples, dim=0)
                                samples = samples.cpu()
                                all_samples.append(samples)

                            print(f"\nMetrics for config: {sampling_config_to_str(sampling_config)}:")
                            for i in range(len(datasets)):
                                metrics = score_sudoku(all_samples[i], all_diffusion_mask[i], all_solutions_tokenized[i], tokenizer)
                                print(f"Dataset: {ds_names[i]}")
                                for key, value in metrics.items():
                                    print(" " * 2 + f"{key}: {value:.4f}")

                        if position_sampling_strategy != "top_k_gumbel":
                            break
                    if position_sampling_strategy != "top_k_gumbel":
                        break
                if position_sampling_strategy != "top_k_gumbel":
                    break


if __name__ == "__main__":
    main()