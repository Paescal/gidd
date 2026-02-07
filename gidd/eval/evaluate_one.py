import argparse
import hydra
import tqdm
import torch
import numpy as np
import os
import csv
import time

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

def get_info_to_collect(strategy, beam_search_config: bool):
    if beam_search_config.do_beam_search == 'true':
        # return ["prune_correct", "beam_search_forward_calls", "beam_search_branch_correctness"]
        return ["prune_correct", "beam_search_forward_calls", "beam_search_beam_correctness"]
        # return ["prune_correct", "beam_search_forward_calls"]
    if strategy in ["mdlm_vanilla", "mdlm_adaptive_score_select_update"]:
        return ["history", "logits", "forward_calls"]
    elif strategy in ["gidd_prob_to_recover_data"]:
        # return ["history", "logits", "confidence_t_0", "marginals", "change_events", "forward_calls"]
        return ["history", "logits", "marginals", "change_events", "forward_calls"]
    else:
        # return ["history", "logits", "confidence_t_0", "marginals", "change_events", "forward_calls"]
        return ["history", "logits", "marginals", "change_events", "forward_calls"]

def main(sampling_config):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    seed = sampling_config.general_sampling.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_path = hydra.utils.to_absolute_path(f"./outputs/{sampling_config.general_sampling.checkpoint}")
    model, noise_schedule, ckpt_tokenizer, ckpt_config = load_checkpoint(model_path, device=device)
    dtype = parse_dtype(ckpt_config.training.dtype)
    ds_path = hydra.utils.to_absolute_path(f"./gidd/datasets/sudoku_shah/{sampling_config.general_sampling.dataset}/test")
    ds = load_from_disk(ds_path)
    data_loader = _get_dataloader_with_seed(seed, ckpt_config, ds, shuffle=True, drop_last=False, batch_size=sampling_config.general_sampling.batch_size, collate_fn=partial(default_collator, ckpt_config, ckpt_tokenizer, text_key="text"), persistent_workers=False)
    sampler = get_sampler(ckpt_config, model, ckpt_tokenizer, noise_schedule, sampling_config=sampling_config.sampling_strategy, min_p=args.min_p)
    
    model.eval()
    strategy_metrics = {}
    info_to_collect = get_info_to_collect(sampling_config.sampling_strategy.strategy, sampling_config.beam_search)
    start_time = time.time()
    with tqdm.tqdm(total=sampling_config.general_sampling.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            data_loader = iter(data_loader)
            generation_info_handler = GenerationInfoHandler(info=info_to_collect, max_seq_len=ckpt_config.model.max_seq_len, sampling_config_dict=namespace_to_dict(sampling_config), noise_schedule=noise_schedule)
            for i in range(0, sampling_config.general_sampling.num_samples, sampling_config.general_sampling.batch_size):
                batch = next(data_loader)
                bs = min(sampling_config.general_sampling.batch_size, sampling_config.general_sampling.num_samples - i)
                batch = batch[:bs]
                diffusion_mask = batch['diffusion_mask']
                solutions_tokenized = batch['input_ids']
                if ckpt_config.training.use_diffusion_mask:
                    if i > 0:
                        generation_info_handler.collect_history = False
                        generation_info_handler.collect_logits = False
                        generation_info_handler.collect_confidence_t_0 = False
                        generation_info_handler.collect_marginals = False
                        generation_info_handler.collect_prune_correct = False
                    if i >= 2 * sampling_config.general_sampling.batch_size:
                        generation_info_handler.collect_change_events = False

                    samples = sampler.generate_from_given(batch['puzzle_ids'], diffusion_mask, solution=solutions_tokenized, num_denoising_steps=sampling_config.general_sampling.num_denoising_steps, num_self_correction_steps=sampling_config.general_sampling.num_self_correction_steps, decode=False, show_progress=False, generation_info_handler=generation_info_handler, beam_search_config=sampling_config.beam_search)

                if ckpt_config.model.puzzle_conditioning == 'in_context':
                        samples = samples[..., -ckpt_config.model.max_seq_len:]
                        diffusion_mask = diffusion_mask[..., -ckpt_config.model.max_seq_len:]
                        solutions_tokenized = solutions_tokenized[..., -ckpt_config.model.max_seq_len:]
                
                batch_metrics = score_sudoku(samples.cpu(), diffusion_mask, solutions_tokenized, ckpt_tokenizer)
                for k, v in batch_metrics.items():
                    strategy_metrics[k] = strategy_metrics.get(k, 0) + v * bs
                pbar.update(bs)
    end_time = time.time()
    time_taken = end_time - start_time
    accuracy = strategy_metrics['correct_solution'].item() / sampling_config.general_sampling.num_samples
    correctly_filled_cells = strategy_metrics['correctly_filled_cells'].item() / sampling_config.general_sampling.num_samples
    not_fully_unmasked = strategy_metrics['not_fully_unmasked'].item() / sampling_config.general_sampling.num_samples
    print(f"accuracy={accuracy:.4f}")
    print(f"correctly_filled_cells={correctly_filled_cells:.4f}")
    print(f"not_fully_unmasked={not_fully_unmasked:.4f}")
    if sampling_config.beam_search.do_beam_search == 'true':
        beam_search_fwd_calls = generation_info_handler.get_beam_search_forward_calls()
        nfe = torch.mean(beam_search_fwd_calls.to(dtype=torch.float32)).item()
        print(f"NFE: {nfe:.2f} forward calls per sample with beam search")
    else:
        forward_calls = generation_info_handler.get_forward_calls()["forward_calls_by_batch"].sum().item()
        nfe = forward_calls / sampling_config.general_sampling.num_samples
        print(f"NFE: {nfe:.2f} forward calls per sample")
    print(f"Time taken: {time_taken:.0f} seconds for {sampling_config.general_sampling.num_samples} samples")
    speed = sampling_config.general_sampling.num_samples / time_taken
    print(f"Speed: {speed:.2f} samples/second")

    generation_info_handler.collect_history = "history" in info_to_collect
    generation_info_handler.collect_logits = "logits" in info_to_collect
    generation_info_handler.collect_confidence_t_0 = "confidence_t_0" in info_to_collect
    generation_info_handler.collect_marginals = "marginals" in info_to_collect
    generation_info_handler.collect_change_events = "change_events" in info_to_collect
    generation_info_handler.collect_forward_calls = "forward_calls" in info_to_collect
    generation_info_handler.collect_prune_correct = "prune_correct" in info_to_collect
    generation_info_handler.save_info(
        f"/local/home/prisold/gidd/outputs/generation_info/combination_{sampling_config.general_sampling.combinations_row}",
        meta={
            "accuracy": accuracy,
            "correctly_filled_cells": correctly_filled_cells,
            "nfe": nfe,
            "time_taken_s": time_taken,
            "speed_samples_per_s": speed,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general_sampling_argument_group = parser.add_argument_group('General sampling arguments')
    general_sampling_argument_group.add_argument('--device', type=int, default=0, help='Device to use for evaluation') # Not actually used!
    general_sampling_argument_group.add_argument('--seed', type=int, default=1, help='Seed for reproducibility')
    general_sampling_argument_group.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to evaluate')
    general_sampling_argument_group.add_argument('--dataset', type=str, required=True, help='Dataset to evaluate (easy, hard)')
    general_sampling_argument_group.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    general_sampling_argument_group.add_argument('--num_denoising_steps', type=int, default=81, help='Number of denoising steps')
    general_sampling_argument_group.add_argument('--num_self_correction_steps', type=int, default=81, help='Number of self-correction steps')
    general_sampling_argument_group.add_argument('--batch_size', type=int, default=64, help='Batch size')
    general_sampling_argument_group.add_argument('--min_p', type=float, default=0, help='Minimum probability to be chosen in categorical sampling')
    general_sampling_argument_group.add_argument('--combinations_row', type=int, default=0, help='Row id for combinations file of current combination')

    sampling_strategy_argument_group = parser.add_argument_group('Sampling arguments')
    sampling_strategy_argument_group.add_argument('--strategy', type=str, required=True, help='Sampling strategy to evaluate')
    sampling_strategy_argument_group.add_argument('--time_steps', type=str, default='fixed', help='Fixed or inferred time steps')
    sampling_strategy_argument_group.add_argument('--score_mask_position', type=str, default=None, help='Scoring function for unmasking a position')
    sampling_strategy_argument_group.add_argument('--score_position_for_change', type=str, default=None, help='Scoring function for changing the token at a position')
    sampling_strategy_argument_group.add_argument('--select_position', type=str, default=None, help='Sampling strategy for selecting a position')
    sampling_strategy_argument_group.add_argument('--select_position_change', type=str, default=None, help='Sampling strategy for selecting a position when changing an unmasked token')
    sampling_strategy_argument_group.add_argument('--select_position_unmask', type=str, default=None, help='Sampling strategy for selecting a position when unmasking a token')
    sampling_strategy_argument_group.add_argument('--change_token', type=str, default=None, help='Sampling strategy for updating a token when changing an unmasked token')
    sampling_strategy_argument_group.add_argument('--unmask_token', type=str, default=None, help='Sampling strategy for updating a token when unmasking a token')
    sampling_strategy_argument_group.add_argument('--k', type=int, default=None, help='K for top-k gumbel sampling')
    sampling_strategy_argument_group.add_argument('--gumbel_noise_coefficient', type=float, default=None, help='Gumbel noise coefficient for top-k gumbel sampling')
    sampling_strategy_argument_group.add_argument('--self_correction', type=str, default="none", help='Self-correction strategy to use')
    sampling_strategy_argument_group.add_argument('--oracle', type=str, default="model", help='Oracle for p_denoise')
    sampling_strategy_argument_group.add_argument('--position_sampling', type=str, default="independent", help='Position sampling strategy for p_denoise')
    sampling_strategy_argument_group.add_argument('--position_metric', type=str, default="p_denoise", help='Position metric for p_denoise')
    sampling_strategy_argument_group.add_argument('--token_sampling', type=str, default="categorical", help='Token sampling strategy for p_denoise')
    sampling_strategy_argument_group.add_argument('--uniform_noise', type=str, default="none", help='Uniform noise strategy for p_denoise')

    beam_search_argument_group = parser.add_argument_group('Beam search arguments')
    beam_search_argument_group.add_argument('--do_beam_search', type=str, default='false', help='Whether to use beam search')
    beam_search_argument_group.add_argument('--steps_before_pruning', type=int, default=1, help='Number of steps before pruning beams')
    beam_search_argument_group.add_argument('--pruning_num_beams', type=int, default=1, help='Number of beams to prune at each pruning step')
    beam_search_argument_group.add_argument('--branching_factor', type=int, default=2, help='Branching factor for beam search')
    beam_search_argument_group.add_argument('--score_time', type=str, default='after_pruning', help='The time to use in beam scoring for pruning')
    beam_search_argument_group.add_argument('--score_method', type=str, default='avg', help='Method to score beams for pruning')
    beam_search_argument_group.add_argument('--initiate_beam_search_after_progress', type=float, default=0.1, help='Progress threshold to initiate beam search')

    args = parser.parse_args()

    sampling_config = dict_to_namespace({
        "general_sampling":{
            "device": args.device,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "num_denoising_steps": args.num_denoising_steps,
            "num_self_correction_steps": args.num_self_correction_steps,
            "batch_size": args.batch_size,
            "min_p": args.min_p,
            "combinations_row": args.combinations_row,
        },
        "sampling_strategy": {
            "strategy": args.strategy,
            "time_steps": args.time_steps,
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
            },
        },
        "beam_search": {
            "do_beam_search": args.do_beam_search,
            "steps_before_pruning": args.steps_before_pruning,
            "pruning_num_beams": args.pruning_num_beams,
            "branching_factor": args.branching_factor,
            "score_time": args.score_time,
            "score_method": args.score_method,
            "initiate_beam_search_after_progress": args.initiate_beam_search_after_progress,
        },
    })

    main(sampling_config)