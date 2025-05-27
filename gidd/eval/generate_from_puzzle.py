import hydra
import tqdm
import torch
import numpy as np
import os

from functools import partial
from pathlib import Path
from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler
from datasets import load_from_disk

@hydra.main(config_path="../configs", config_name="generate_from_puzzle", version_base="1.1")
def main(config):
# args pass batch size, ckpt_path, num_denoising_steps and min_p
    def add_bos_eos(examples, cols):
        for col in cols:
            examples[col] = [tokenizer.bos_token + example + tokenizer.eos_token for example in examples[col]]
        return examples
    
    num_samples = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)

    model, noise_schedule, tokenizer, ckpt_config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    ckpt_config.training.eval_batch_size = config.batch_size
    dtype = parse_dtype(ckpt_config.training.dtype)

    # ds_eval = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/{ckpt_config.data.dataset_name}{('_' + ckpt_config.data.dataset_subset) if ckpt_config.data.dataset_subset else ''}/evaluate")
    ds_eval_easy = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/easy/test")
    ds_eval_hard = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/hard/test")
    datasets = {"easy": ds_eval_easy, "hard": ds_eval_hard}
    for ds_name, ds_eval in datasets.items():
        ds_eval = ds_eval.select(range(num_samples))
        
        print(f"Evaluating model from {ckpt_path} on {num_samples} samples")
        
        # ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle', 'solution']), batched=True)
        # ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle', 'text']), batched=True)
        puzzles = ds_eval.select_columns(['puzzle'])
        # solutions = ds_eval.select_columns(['solution'])
        solutions = ds_eval.select_columns(['text'])

        puzzles_tokenized = torch.tensor(tokenizer(puzzles['puzzle'])['input_ids'])
        # solutions_tokenized = torch.tensor(tokenizer(solutions['solution'])['input_ids'])
        solutions_tokenized = torch.tensor(tokenizer(solutions['text'])['input_ids'])
        diffusion_mask = (puzzles_tokenized == tokenizer.mask_token_id).to(int)
        # print(f"diffusion_mask: {diffusion_mask}")
        sampler = get_sampler(ckpt_config, model, tokenizer, noise_schedule, sampling_config=config, compile_step=config.compilation.compile_torch, min_p=config.min_p)
        model.eval()

        samples = []
        with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                for i in range(0, num_samples, config.batch_size):
                    bs = min(config.batch_size, num_samples - i)
                    # TODO: how is the max_length in SamplerInstance.model.config.max_seq_len set? Once that is done automatically for sudoku, no need to pass it here
                    if ckpt_config.training.use_diffusion_mask:
                        z_t = sampler.generate_from_given(puzzles_tokenized[i:i+bs], diffusion_mask[i:i+bs], config.num_denoising_steps, max_length=ckpt_config.model.max_seq_len, decode=False, show_progress=False, keep_history=False)
                    else:
                        z_t = sampler.generate_from_expected_t(puzzles_tokenized[i:i+bs], diffusion_mask[i:i+bs], config.num_denoising_steps, add_random_tokens=(ckpt_config.model.p_uniform == 0), max_length=ckpt_config.model.max_seq_len, decode=False, show_progress=False, keep_history=False)
                    samples.append(z_t)
                    pbar.update(bs)
        samples = torch.cat(samples, dim=0)

        # samples = samples.cpu()[:, 1:-1]
        # diffusion_mask = diffusion_mask.cpu()[:, 1:-1]
        samples = samples.cpu()
        diffusion_mask = diffusion_mask.cpu()

        samples_dir = os.path.join(ckpt_path, f"../../samples/{ds_name}/")
        Path(samples_dir).mkdir(parents=True, exist_ok=True)
        pre_correction_samples_path = os.path.join(samples_dir, "evaluation_samples_pre_correction.pt")
        diffusion_mask_path = os.path.join(samples_dir, "evaluation_diffusion_mask.pt")
        solutions_path = os.path.join(samples_dir, "evaluation_solutions.pt")

        assert(samples.shape == diffusion_mask.shape and samples.shape == solutions_tokenized.shape)
        print(f"Samples shape: {samples.shape}")
        torch.save(samples, hydra.utils.to_absolute_path(pre_correction_samples_path))
        torch.save(diffusion_mask, hydra.utils.to_absolute_path(diffusion_mask_path))
        torch.save(solutions_tokenized, hydra.utils.to_absolute_path(solutions_path))


if __name__ == "__main__":
    main()