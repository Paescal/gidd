import hydra
import tqdm
import torch
import numpy as np
import os
import csv

from functools import partial
from pathlib import Path
from gidd.utils import parse_dtype, score_sudoku
from gidd.checkpoints import load_checkpoint
from gidd.data import _get_dataloader, default_collator
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
    # ds_train = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/easy/train")
    ds_eval_easy = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/easy/test")
    ds_eval_hard = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/hard/test")
    datasets = {
        # "train": ds_train,
        "easy": ds_eval_easy,
        "hard": ds_eval_hard
        }

    ds_names = list(datasets.keys())

    data_loaders = []
    for ds in datasets.values():
        ds_data_loaders = []
        for ckpt_config in ckpt_configs:
            ds_data_loader = _get_dataloader(ckpt_config, ds, shuffle=False, drop_last=False, batch_size=config.batch_size, collate_fn=partial(default_collator, ckpt_config, tokenizer, text_key="text"))
            ds_data_loaders.append(ds_data_loader)
        data_loaders.append(ds_data_loaders)

    all_puzzles_tokenized = []
    all_solutions_tokenized = []
    all_diffusion_masks = []
    for i in range(len(datasets)):
        for data_loader in data_loaders[i]:
            puzzles_tokenized = []
            solutions_tokenized = []
            diffusion_mask = []
            for j in range(0, num_samples, config.batch_size):
                batch = next(iter(data_loader))
                puzzles_tokenized.append(batch['puzzle_ids'])
                solutions_tokenized.append(batch['input_ids'])
                diffusion_mask.append(batch['diffusion_mask'])
            all_puzzles_tokenized.append(torch.cat(puzzles_tokenized, dim=0))
            all_solutions_tokenized.append(torch.cat(solutions_tokenized, dim=0))
            all_diffusion_masks.append(torch.cat(diffusion_mask, dim=0))

        all_puzzles_tokenized.append(puzzles_tokenized)
        all_solutions_tokenized.append(solutions_tokenized)
        all_diffusion_masks.append(diffusion_mask)

    position_metrics = config.sampling.position_metric
    position_sampling_strategies = config.sampling.position_sampling_strategy
    top_k_gumbel_k = config.sampling.position_sampling_strategy_args.top_k_gumbel.k
    top_k_gumbel_noise_coefficient = config.sampling.position_sampling_strategy_args.top_k_gumbel.gumbel_noise_coefficient
    token_sampling_strategies = config.sampling.token_sampling_strategy

    csv_rows = []
    for model, noise_schedule, ckpt_config, ckpt_path in zip(models, noise_schedules, ckpt_configs, ckpt_paths):
        print(f"\n\n{'-'*10}Evaluating model from {ckpt_path} on {num_samples} samples{'-'*10}\n")
        ckpt_parts = Path(ckpt_path).parts
        idx = ckpt_parts.index("outputs")
        model_name = "/".join(ckpt_parts[idx + 1:idx + 3])

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
                                                z_t = sampler.generate_from_given(all_puzzles_tokenized[i][j:j+bs], all_diffusion_masks[i][j:j+bs], num_denoising_steps=config.num_denoising_steps, decode=False, show_progress=False, keep_history=False)
                                            else:
                                                z_t = sampler.generate_from_expected_t(all_puzzles_tokenized[i][j:j+bs], all_diffusion_masks[i][j:j+bs], config.num_denoising_steps, add_random_tokens=(ckpt_config.model.p_uniform == 0), decode=False, show_progress=False, keep_history=False)
                                            samples.append(z_t)
                                            pbar.update(bs)
                                samples = torch.cat(samples, dim=0)
                                samples = samples.cpu()
                                all_samples.append(samples)

                            print(f"\nMetrics for config: {sampling_config_to_str(sampling_config)}:")
                            for i in range(len(datasets)):
                                metrics = score_sudoku(all_samples[i], all_diffusion_masks[i], all_solutions_tokenized[i], tokenizer)
                                print(f"Dataset: {ds_names[i]}")
                                for key, value in metrics.items():
                                    print(" " * 2 + f"{key}: {value:.4f}")
                                
                                csv_rows.append({
                                    'model': model_name,
                                    'accuracy': metrics.get('correct_solution', float('nan')).item(),
                                    'dataset': ds_names[i],
                                    'position_metric': position_metric,
                                    'position_strategy': position_sampling_strategy,
                                    'token_strategy': token_sampling_strategy,
                                    'k': k,
                                    'gumbel': noise_coefficient
                                })
                        if position_sampling_strategy != "top_k_gumbel":
                            break
                    if position_sampling_strategy != "top_k_gumbel":
                        break
                if position_sampling_strategy != "top_k_gumbel":
                    break
    
    output_dir = "/local/home/prisold/gidd/outputs/evaluation_logs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "latest.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ['model', 'accuracy', 'dataset', 'position_metric', 'position_strategy', 'token_strategy', 'k', 'gumbel']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()