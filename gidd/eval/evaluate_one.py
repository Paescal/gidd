import argparse
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

def main(args, sampling_config):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_path = hydra.utils.to_absolute_path(f"./outputs/{args.checkpoint}")
    model, noise_schedule, ckpt_tokenizer, ckpt_config = load_checkpoint(model_path, device=device)
    dtype = parse_dtype(ckpt_config.training.dtype)
    ds_path = hydra.utils.to_absolute_path(f"./gidd/datasets/sudoku_shah/{args.dataset}/test")
    ds = load_from_disk(ds_path)
    data_loader = _get_dataloader(ckpt_config, ds, shuffle=False, drop_last=False, batch_size=args.batch_size, collate_fn=partial(default_collator, ckpt_config, ckpt_tokenizer, text_key="text"), persistent_workers=False)
    sampler = get_sampler(ckpt_config, model, ckpt_tokenizer, noise_schedule, sampling_config=sampling_config, compile_step=bool(args.compile_torch), min_p=args.min_p)
    
    model.eval()
    strategy_metrics = {}
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            data_loader = iter(data_loader)
            for i in range(0, args.num_samples, args.batch_size):
                batch = next(data_loader)
                bs = min(args.batch_size, args.num_samples - i)
                batch = batch[:bs]
                diffusion_mask = batch['diffusion_mask']
                solutions_tokenized = batch['input_ids']
                if ckpt_config.training.use_diffusion_mask:
                    samples = sampler.generate_from_given(batch['puzzle_ids'], diffusion_mask, num_denoising_steps=args.num_denoising_steps, decode=False, show_progress=False, keep_history=False)
                try:
                    if ckpt_config.model.puzzle_conditioning == 'in_context':
                            samples = samples[..., -ckpt_config.model.max_seq_len:]
                            diffusion_mask = diffusion_mask[..., -ckpt_config.model.max_seq_len:]
                            solutions_tokenized = solutions_tokenized[..., -ckpt_config.model.max_seq_len:]
                except:
                    pass
                batch_metrics = score_sudoku(samples.cpu(), diffusion_mask, solutions_tokenized, ckpt_tokenizer)
                for k, v in batch_metrics.items():
                    strategy_metrics[k] = strategy_metrics.get(k, 0) + v * bs
                pbar.update(bs)
    accuracy = strategy_metrics['correct_solution'].item() / args.num_samples
    correctly_filled_cells = strategy_metrics['correctly_filled_cells'].item() / args.num_samples
    not_fully_unmasked = strategy_metrics['not_fully_unmasked'].item() / args.num_samples
    print(f"accuracy={accuracy:.4f}")
    print(f"correctly_filled_cells={correctly_filled_cells:.4f}")
    print(f"not_fully_unmasked={not_fully_unmasked:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Device to use for evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate (easy, hard)')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--num_denoising_steps', type=int, default=81, help='Number of denoising steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--min_p', type=float, default=0, help='Minimum probability to be chosen in categorical sampling')
    parser.add_argument('--compile_torch', type=int, default=False, help='Whether to compile the torch model')
    
    sampling_argument_group = parser.add_argument_group('Sampling arguments')
    sampling_argument_group.add_argument('--strategy', type=str, required=True, help='Sampling strategy to evaluate')
    sampling_argument_group.add_argument('--score_position', type=str, default=None, help='Scoring function for updating a position')
    sampling_argument_group.add_argument('--select_position', type=str, default=None, help='Sampling strategy for selecting a position')
    sampling_argument_group.add_argument('--select_position_change', type=str, default=None, help='Sampling strategy for selecting a position when changing an unmasked token')
    sampling_argument_group.add_argument('--select_position_unmask', type=str, default=None, help='Sampling strategy for selecting a position when unmasking a token')
    sampling_argument_group.add_argument('--update_token', type=str, default=None, help='Sampling strategy for updating a token')
    sampling_argument_group.add_argument('--update_token_change', type=str, default=None, help='Sampling strategy for updating a token when changing an unmasked token')
    sampling_argument_group.add_argument('--update_token_unmask', type=str, default=None, help='Sampling strategy for updating a token when unmasking a token')
    sampling_argument_group.add_argument('--k', type=int, default=None, help='K for top-k gumbel sampling')
    sampling_argument_group.add_argument('--gumbel_noise_coefficient', type=float, default=None, help='Gumbel noise coefficient for top-k gumbel sampling')

    args = parser.parse_args()

    sampling_config = dict_to_namespace({
        "sampling": {
            "strategy": args.strategy,
            "score_position": args.score_position,
            "select_position": args.select_position,
            "select_position_change": args.select_position_change,
            "select_position_unmask": args.select_position_unmask,
            "update_token": args.update_token,
            "update_token_change": args.update_token_change,
            "update_token_unmask": args.update_token_unmask,
            "k": args.k,
            "gumbel_noise_coefficient": args.gumbel_noise_coefficient
        }
    })

    main(args, sampling_config)