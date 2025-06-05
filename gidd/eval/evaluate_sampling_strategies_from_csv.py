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

@hydra.main(config_path="../configs", config_name="evaluate_sampling_strategies", version_base="1.1")
def main(config):
# args pass batch size, ckpt_path, num_denoising_steps and min_p

    num_samples = 6400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    input_dir = hydra.utils.to_absolute_path("./gidd/configs/sampling/")
    os.makedirs(input_dir, exist_ok=True)
    strategies_path = os.path.join(input_dir, "strategies.csv")
    strategies = []
    metrics = []
    models = {}
    datasets = {}
    tokenizer = None
    dtype = None
    with open(strategies_path, "r", newline="") as strategies_file:
        reader = csv.DictReader(strategies_file)
        for row in reader:
            model_id = row['model']
            if model_id not in models:
                model_path = hydra.utils.to_absolute_path(f"./outputs/{model_id}/checkpoints/latest/")
                model, noise_schedule, ckpt_tokenizer, ckpt_config = load_checkpoint(model_path, device=device)
                if tokenizer is None:
                    tokenizer = ckpt_tokenizer
                if dtype is None:
                    dtype = parse_dtype(ckpt_config.training.dtype)
                models[model_id] = {
                    'model': model,
                    'noise_schedule': noise_schedule,
                    'tokenizer': ckpt_tokenizer,
                    'config': ckpt_config
                }
            else:
                model = models[model_id]['model']
                noise_schedule = models[model_id]['noise_schedule']
                ckpt_config = models[model_id]['config']
            
            ds_name = row['dataset']
            if ds_name not in datasets:
                ds_path = hydra.utils.to_absolute_path(f"./gidd/datasets/sudoku_shah/{ds_name}/test")
                ds = load_from_disk(ds_path)
                datasets[ds_name] = ds
            else:
                ds = datasets[ds_name]
            
            data_loader = _get_dataloader(ckpt_config, ds, shuffle=False, drop_last=False, batch_size=config.batch_size, collate_fn=partial(default_collator, ckpt_config, tokenizer, text_key="text"), persistent_workers=False)

            strategies.append({
                'num_denoising_steps': int(row['num_denoising_steps']),
                'model': model_id,
                'dataset': ds_name,
                'data_loader': data_loader,
                'position_metric': row['position_metric'],
                'position_sampling_strategy': row['position_strategy'],
                'token_sampling_strategy': row['token_strategy'],
                'top_k_gumbel_k': int(row['k']),
                'top_k_gumbel_noise_coefficient': float(row['gumbel']),
            })
    with tqdm.tqdm(total=len(strategies), desc="Evaluating strategies", dynamic_ncols=True) as strategy_pbar:
        for strategy in strategies:
            sampling_config = {
                "sampling": {
                    "position_metric": strategy['position_metric'],
                    "position_sampling_strategy": strategy['position_sampling_strategy'],
                    "position_sampling_strategy_args": {
                        "top_k_gumbel": {
                            "k": strategy['top_k_gumbel_k'],
                            "gumbel_noise_coefficient": strategy['top_k_gumbel_noise_coefficient']
                        }
                    },
                    "token_sampling_strategy": strategy['token_sampling_strategy']
                }
            }
            sampling_config = dict_to_namespace(sampling_config)

            model = models[strategy['model']]['model']
            noise_schedule = models[strategy['model']]['noise_schedule']
            ckpt_config = models[strategy['model']]['config']

            sampler = get_sampler(ckpt_config, model, tokenizer, noise_schedule, sampling_config=sampling_config, compile_step=config.compilation.compile_torch, min_p=config.min_p)
            model.eval()

            samples = []
            data_loader = strategy['data_loader']
            strategy_metrics = {}
            with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
                with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                    data_loader = iter(data_loader)
                    for i in range(0, num_samples, config.batch_size):
                        batch = next(data_loader)
                        bs = min(config.batch_size, num_samples - i)
                        batch = batch[:bs]
                        if ckpt_config.training.use_diffusion_mask:
                            z_t = sampler.generate_from_given(batch['puzzle_ids'], batch['diffusion_mask'], num_denoising_steps=strategy['num_denoising_steps'], decode=False, show_progress=False, keep_history=False)
                        batch_metrics = score_sudoku(z_t.cpu(), batch['diffusion_mask'], batch['input_ids'], tokenizer)
                        for k, v in batch_metrics.items():
                            strategy_metrics[k] = strategy_metrics.get(k, 0) + v * bs
                        pbar.update(bs)
            strategy['data_loader'] = None
            metrics.append(strategy_metrics)
            strategy_pbar.update(1)
    
    output_dir = hydra.utils.to_absolute_path("./outputs/evaluation_logs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"latest_{num_samples}.csv")
    with open(output_path, "w", newline="") as out_file:
        fieldnames = ['accuracy', 'model', 'dataset', 'position_metric', 'position_strategy', 'token_strategy', 'k', 'gumbel']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, strategy in enumerate(strategies):
            strategy_metrics = metrics[i]
            row = {
                'accuracy': strategy_metrics['correct_solution'].item() / num_samples,
                'num_denoising_steps': strategy['num_denoising_steps'],
                'model': strategy['model'],
                'dataset': strategy['dataset'],
                'position_metric': strategy['position_metric'],
                'position_strategy': strategy['position_sampling_strategy'],
                'token_strategy': strategy['token_sampling_strategy'],
                'k': strategy['top_k_gumbel_k'],
                'gumbel': strategy['top_k_gumbel_noise_coefficient']
            }
            writer.writerow(row)


if __name__ == "__main__":
    main()