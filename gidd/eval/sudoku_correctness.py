from transformers import AutoTokenizer
from datasets import load_from_disk

# Evaluation for samples from pure noise

def correct_cells_score(samples, solutions_tokenized):
    correct_cells = (solutions_tokenized == samples).to(int)
    return torch.sum(correct_cells, dim=-1)

# For each row, column and sub-square, check whether the numbers 1 to sudoku_size (e.g. 9 for 9x9) are present. If so, add 1 to the score.
# Expects the sudoku to be a 2D array of shape (sudoku_size, sudoku_size).
def row_col_box_set_score(sudoku):
    row_sets = [set(row) for row in sudoku]
    col_sets = [set(col) for col in sudoku.T]
    box_size = int(len(sudoku) ** 0.5)
    box_sets =[set(sudoku[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]) for i in range(box_size) for j in range(box_size)]
    score = 0
    for curr_set in row_sets + col_sets + box_sets:
        for i in range(len(sudoku)):
            if str(i + 1) in curr_set:
                score += 1
    return score


def count_unique_samples(samples):
    unique_samples = set()
    for sample in samples:
        # Convert the sample to a tuple (or any hashable type) to add it to the set
        unique_samples.add(tuple(sample))
    return len(unique_samples)

def score_samples(samples, solutions_tokenized, tokenizer):
    cells_score = correct_cells_score(samples, solutions_tokenized)
    seq_len = samples.shape[-1]
    print(f"Number of correct cells (max is {seq_len}): {cells_score}")
    print(f"Average number of correct cells: {torch.mean(cells_score.float())}")
    print(f"number of fully correct samples: {torch.sum(cells_score == (seq_len))}")

    samples_decoded = np.array([tokenizer.decode(samples[i], skip_special_tokens=False, clean_up_tokenization_spaces=False).split() for i in range(len(samples))])

    num_unique_samples = count_unique_samples(samples_decoded)
    print(f"Number of unique samples (of {len(samples_decoded)} total): {num_unique_samples}")

    sudoku_size = int(seq_len ** 0.5)
    samples_decoded = samples_decoded.reshape((-1, sudoku_size, sudoku_size))

    set_score = [row_col_box_set_score(sample) for sample in samples_decoded]
    print(f"Set scores (max is {sudoku_size * sudoku_size * 2}): {set_score}")
    print(f"Average set score: {np.mean(set_score)}")

# Evaluation for samples starting from puzzles
import hydra
import tqdm
import torch
import numpy as np

from functools import partial
from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler


@hydra.main(config_path="../configs", config_name="generate_from_puzzle", version_base="1.1")
def main(args):

    def add_bos_eos(examples, cols):
        for col in cols:
            examples[col] = [tokenizer.bos_token + example + tokenizer.eos_token for example in examples[col]]
        return examples
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.checkpoint_path)

    model, noise_schedule, tokenizer, ckpt_config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    ckpt_config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(ckpt_config.training.dtype)

    ds_eval = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/{ckpt_config.data.dataset_name}{('_' + ckpt_config.data.dataset_subset) if ckpt_config.data.dataset_subset else ''}/evaluate")
    ds_eval = ds_eval.select(range(16))
    num_samples = ds_eval.num_rows
    
    print(f"Evaluating model from {args.checkpoint_path} on {num_samples} samples")
    
    # ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle', 'solution']), batched=True)
    ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle']), batched=True)
    puzzles = ds_eval.select_columns(['puzzle'])
    solutions = ds_eval.select_columns(['solution'])

    puzzles_tokenized = torch.tensor(tokenizer(puzzles['puzzle'])['input_ids'])
    solutions_tokenized = torch.tensor(tokenizer(solutions['solution'])['input_ids'])
    diffusion_mask = (puzzles_tokenized == tokenizer.mask_token_id).to(int)
    # TODO: set compile_step based on something
    # TODO: pass model and sampling configs
    sampler = get_sampler(ckpt_config, model, tokenizer, noise_schedule, compile_step=False, min_p=args.min_p)
    model.eval()

    samples = []
    with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, num_samples, args.batch_size):
                bs = min(args.batch_size, num_samples - i)
                # TODO: how is the max_length in SamplerInstance.model.config.max_seq_len set? Once that is done automatically for sudoku, no need to pass it here
                # TODO: add parameter to return the generation history
                z_t = sampler.generate_from_given(puzzles_tokenized[i:i+bs], diffusion_mask[i:i+bs], args.num_denoising_steps, max_length=ckpt_config.model.max_seq_len, decode=False, show_progress=False)
                samples.append(z_t)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0)
    post_correction_samples = samples.clone()

    samples = samples.cpu()[:, 1:-1]
    torch.save(samples, hydra.utils.to_absolute_path(ckpt_path + "../../samples/" + "evaluation_samples_pre_correction.pt"))

    score_samples(samples, solutions_tokenized, tokenizer)

    # with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
    #     with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
    #         for i in range(0, num_samples, args.batch_size):
    #             bs = min(args.batch_size, num_samples - i)
    #             # TODO: run self-correction on the samples



if __name__ == "__main__":
    main()