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
from gidd.data import _get_dataloader_with_seed, _get_dataloader, default_collator
from gidd.sampling import get_sampler
from gidd.eval.generation_info import GenerationInfoHandler
from gidd.eval.visualize import history_to_str
from datasets import load_from_disk
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(i) for i in ns]
    else:
        return ns

def get_info_to_collect(strategy):
    if strategy in ["mdlm_vanilla", "mdlm_adaptive_score_select_update"]:
        return ["history", "logits", "forward_calls"]
    elif strategy in ["gidd_prob_to_recover_data"]:
        return ["history", "logits", "confidence_t_0", "marginals", "change_events", "forward_calls"]
    else:
        return ["history", "logits", "confidence_t_0", "marginals", "change_events", "forward_calls"]

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
    data_loader = _get_dataloader_with_seed(seed, ckpt_config, ds, shuffle=True, drop_last=False, batch_size=args.batch_size, collate_fn=partial(default_collator, ckpt_config, ckpt_tokenizer, text_key="text"), persistent_workers=False)
    # data_loader = _get_dataloader(ckpt_config, ds, shuffle=False, drop_last=False, batch_size=args.batch_size, collate_fn=partial(default_collator, ckpt_config, ckpt_tokenizer, text_key="text"), persistent_workers=False)
    sampler = get_sampler(ckpt_config, model, ckpt_tokenizer, noise_schedule, sampling_config=sampling_config, compile_step=bool(args.compile_torch), min_p=args.min_p)
    
    model.eval()
    strategy_metrics = {}
    info_to_collect = get_info_to_collect(sampling_config.sampling.strategy)
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            data_loader = iter(data_loader)
            generation_info_handler = GenerationInfoHandler(info=info_to_collect, max_seq_len=ckpt_config.model.max_seq_len, sampling_config_dict=namespace_to_dict(sampling_config), noise_schedule=noise_schedule)
            for i in range(0, args.num_samples, args.batch_size):
                batch = next(data_loader)
                bs = min(args.batch_size, args.num_samples - i)
                batch = batch[:bs]
                diffusion_mask = batch['diffusion_mask']
                solutions_tokenized = batch['input_ids']
                if ckpt_config.training.use_diffusion_mask:
                    if i > 0:
                        generation_info_handler.collect_history = False
                        generation_info_handler.collect_logits = False
                        generation_info_handler.collect_confidence_t_0 = False
                        generation_info_handler.collect_marginals = False
                    if i >= 2 * args.batch_size:
                        generation_info_handler.collect_change_events = False

                    if i == 0:
                        samples = sampler.generate_from_given(batch['puzzle_ids'], diffusion_mask, solution=solutions_tokenized, num_denoising_steps=args.num_denoising_steps, num_self_correction_steps=args.num_self_correction_steps, decode=False, show_progress=False, generation_info_handler=generation_info_handler, keep_history=True)
                        # samples, history_of_first_batch = sampler.generate_from_given(batch['puzzle_ids'], diffusion_mask, solution=solutions_tokenized, num_denoising_steps=args.num_denoising_steps, num_self_correction_steps=args.num_self_correction_steps, decode=False, show_progress=False, generation_info_handler=generation_info_handler, keep_history=True)
                        # history_solution = solutions_tokenized
                    else:
                        samples = sampler.generate_from_given(batch['puzzle_ids'], diffusion_mask, solution=solutions_tokenized, num_denoising_steps=args.num_denoising_steps, num_self_correction_steps=args.num_self_correction_steps, decode=False, show_progress=False, generation_info_handler=generation_info_handler, keep_history=False)

                if ckpt_config.model.puzzle_conditioning == 'in_context':
                        samples = samples[..., -ckpt_config.model.max_seq_len:]
                        diffusion_mask = diffusion_mask[..., -ckpt_config.model.max_seq_len:]
                        solutions_tokenized = solutions_tokenized[..., -ckpt_config.model.max_seq_len:]
                
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
    
    generation_info_handler.collect_history = "history" in info_to_collect
    generation_info_handler.collect_logits = "logits" in info_to_collect
    generation_info_handler.collect_confidence_t_0 = "confidence_t_0" in info_to_collect
    generation_info_handler.collect_marginals = "marginals" in info_to_collect
    generation_info_handler.collect_change_events = "change_events" in info_to_collect
    generation_info_handler.collect_forward_calls = "forward_calls" in info_to_collect
    generation_info_handler.save_info(f"/local/home/prisold/gidd/outputs/generation_info/combination_{args.combinations_row}")
    # meta, history, marginals, change_events_table = generation_info_handler.load_all("/local/home/prisold/gidd/outputs/generation_info")
    # import pandas as pd
    # if history is not None:
    #     print("history:", history.shape)

    # if marginals is not None:
    #     print("total_accuracy shape:", marginals["total_accuracy"].shape)

    # if isinstance(change_events_table, pd.DataFrame):
    #     print(change_events_table.head())
    #     # Example: conversion rates by step
    #     by_step = (change_events_table.query('event_type == "uniform_to_denoised"')
    #                     .groupby("step").size()
    #             / change_events_table.groupby("step").size())
    #     print("uniform->denoised share by step:\n", by_step)
    
    # generation_info_handler.print_info()
    
    # chosen_sample_for_history = 4 # debugging for checkpoints/gidd_0_2/100_epochs,gidd_keep_where_confident,"score_position_for_change=change_max select_position=top_k_gumbel change_token=change_max k=1 gumbel_noise_coefficient=0 self_correction=none dataset=hard num_samples=64 num_denoising_steps=81 batch_size=64 min_p=0 compile_torch=0 seed=1"
    # chosen_sample_for_history = 32 # debugging for gidd_independent_positions_decomposed_update_distribution vs gidd_emulate_mdlm_vanilla
    # chosen_sample_for_history = 19
    # history = generation_info_handler.get_history().cpu()
    # history_of_chosen_sample = history[chosen_sample_for_history]
    # if ckpt_config.model.puzzle_conditioning == 'in_context':
    #     history_of_chosen_sample = history_of_chosen_sample[:, -ckpt_config.model.max_seq_len:]
    # print(history_to_str(history_of_chosen_sample[:-1], history_of_chosen_sample[-1]))

    # marginals = generation_info_handler.get_marginals()
    # print(marginals)

    # history_of_chosen_sample = history_of_first_batch.cpu()[chosen_sample_for_history]
    # history_solution = history_solution.cpu()[chosen_sample_for_history]
    # if ckpt_config.model.puzzle_conditioning == 'in_context':
    #     history_of_chosen_sample = history_of_chosen_sample[:, -ckpt_config.model.max_seq_len:]
    #     history_solution = history_solution[-ckpt_config.model.max_seq_len:]
    # print(history_to_str(history_of_chosen_sample, history_solution))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='Device to use for evaluation') # Not actually used!
    parser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate (easy, hard)')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--num_denoising_steps', type=int, default=81, help='Number of denoising steps')
    parser.add_argument('--num_self_correction_steps', type=int, default=81, help='Number of self-correction steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--min_p', type=float, default=0, help='Minimum probability to be chosen in categorical sampling')
    parser.add_argument('--compile_torch', type=int, default=False, help='Whether to compile the torch model')
    parser.add_argument('--combinations_row', type=int, default=0, help='Row id for combinations file of current combination')

    sampling_argument_group = parser.add_argument_group('Sampling arguments')
    sampling_argument_group.add_argument('--strategy', type=str, required=True, help='Sampling strategy to evaluate')
    sampling_argument_group.add_argument('--score_mask_position', type=str, default=None, help='Scoring function for unmasking a position')
    sampling_argument_group.add_argument('--score_position_for_change', type=str, default=None, help='Scoring function for changing the token at a position')
    sampling_argument_group.add_argument('--select_position', type=str, default=None, help='Sampling strategy for selecting a position')
    sampling_argument_group.add_argument('--select_position_change', type=str, default=None, help='Sampling strategy for selecting a position when changing an unmasked token')
    sampling_argument_group.add_argument('--select_position_unmask', type=str, default=None, help='Sampling strategy for selecting a position when unmasking a token')
    sampling_argument_group.add_argument('--change_token', type=str, default=None, help='Sampling strategy for updating a token when changing an unmasked token')
    sampling_argument_group.add_argument('--unmask_token', type=str, default=None, help='Sampling strategy for updating a token when unmasking a token')
    sampling_argument_group.add_argument('--k', type=int, default=None, help='K for top-k gumbel sampling')
    sampling_argument_group.add_argument('--gumbel_noise_coefficient', type=float, default=None, help='Gumbel noise coefficient for top-k gumbel sampling')
    sampling_argument_group.add_argument('--self_correction', type=str, default="none", help='Self-correction strategy to use')
    sampling_argument_group.add_argument('--oracle', type=str, default="model", help='Oracle for p_denoise')
    sampling_argument_group.add_argument('--position_sampling', type=str, default="independent", help='Position sampling strategy for p_denoise')
    sampling_argument_group.add_argument('--position_metric', type=str, default="p_denoise", help='Position metric for p_denoise')
    sampling_argument_group.add_argument('--token_sampling', type=str, default="categorical", help='Token sampling strategy for p_denoise')
    sampling_argument_group.add_argument('--uniform_noise', type=str, default="none", help='Uniform noise strategy for p_denoise')

    args = parser.parse_args()

    sampling_config = dict_to_namespace({
        "sampling": {
            "strategy": args.strategy,
            "score_mask_position": args.score_mask_position,
            "score_position_for_change": args.score_position_for_change,
            "select_position": args.select_position,
            "select_position_change": args.select_position_change,
            "select_position_unmask": args.select_position_unmask,
            "change_token": args.change_token,
            "unmask_token": args.unmask_token,
            "k": args.k,
            "gumbel_noise_coefficient": args.gumbel_noise_coefficient,
            "self_correction": args.self_correction,
            "p_denoise": {
                "oracle": args.oracle,
                "position_sampling": args.position_sampling,
                "position_metric": args.position_metric,
                "token_sampling": args.token_sampling,
                "uniform_noise": args.uniform_noise,
            }
        }
    })

    main(args, sampling_config)