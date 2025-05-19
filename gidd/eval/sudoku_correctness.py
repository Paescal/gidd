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
    box_sets =[set(sudoku[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size].flatten()) for i in range(box_size) for j in range(box_size)]
    score = 0
    for curr_set in row_sets + col_sets + box_sets:
        for i in range(len(sudoku)):
            if str(i + 1) in curr_set:
                score += 1
    return score


def count_unique_samples(samples):
    unique_samples = set()
    for sample in samples:
        unique_samples.add(tuple(sample))
    return len(unique_samples)

def score_samples(samples, diffusion_mask, solutions_tokenized, tokenizer, print_lists=False):
    num_samples = samples.shape[0]
    seq_len = samples.shape[-1]
    cells_score = correct_cells_score(samples, solutions_tokenized)
    free_cells = torch.sum((diffusion_mask == 0).to(int), dim=-1)
    cells_score_filled = cells_score - free_cells
    max_cells_score_filled = seq_len - free_cells
    fraction_filled_correctly = cells_score_filled / max_cells_score_filled

    samples_decoded = np.array([tokenizer.decode(samples[i], skip_special_tokens=False, clean_up_tokenization_spaces=False).split() for i in range(len(samples))])

    num_unique_samples = count_unique_samples(samples_decoded)

    sudoku_size = int(seq_len ** 0.5)
    samples_decoded = samples_decoded.reshape((-1, sudoku_size, sudoku_size))

    set_score = [row_col_box_set_score(sample) for sample in samples_decoded]
    max_set_score = sudoku_size * sudoku_size * 3

    if print_lists:
        print(f"Number of correct cells (max is {seq_len}): {cells_score}")
        print(f"Number of provided cells: {free_cells}")
        print(f"Percentage of missing cells filled correctly: {fraction_filled_correctly}")

    print(f"Percentage of missing cells filled correctly over all samples: {torch.sum(cells_score_filled) / torch.sum(max_cells_score_filled):.2%}")
    # print(f"Average number of correct cells: {torch.mean(cells_score.float())} / {seq_len} ({torch.mean(cells_score.float()) / seq_len:.2%})")
    print(f"number of fully correct samples: {torch.sum(cells_score == (seq_len))} / {num_samples} ({torch.sum(cells_score == (seq_len)) / num_samples:.2%})")
    
    print(f"Number of unique samples: {num_unique_samples} / {num_samples} ({num_unique_samples / num_samples:.2%})")

    if print_lists:
        print(f"Set scores (max is {max_set_score}): {set_score}")
    print(f"Average set score: {np.mean(set_score)} / {max_set_score} ({np.mean(set_score) / max_set_score:.2%})")

def show_history(histories, diffusion_mask):
    # histories.shape = (num_samples, num_denoising_steps, seq_len)
    # diffusion_mask.shape = (num_samples, seq_len)

    sample_history = histories[0]       # shape: (num_steps, seq_len)
    changing_mask = diffusion_mask[0]   # shape: (seq_len,)
    
    num_steps, seq_len = sample_history.shape
    changing_indices = [i for i in range(seq_len) if changing_mask[i] == 1]
    # changing_indices = [i for i in range(seq_len)]
    # changing_indices = changing_indices[:30]

    print("History for changing tokens only:")
    print("-" * (10 + len(changing_indices) * 6))

    # Header
    header = f"{'Step':<6} | " + " ".join(f"{i:>4}" for i in changing_indices)
    print(header)
    print("-" * len(header))

    # Rows
    for step in range(num_steps):
        current_tokens = sample_history[step]
        if step == 0:
            line = f"{step:<6} | " + " ".join(f"{current_tokens[i]:>4}" for i in changing_indices)
        else:
            prev_tokens = sample_history[step - 1]
            line = f"{step:<6} | " + " ".join(
                f"{current_tokens[i]:>4}" if current_tokens[i] != prev_tokens[i] else "  . "
                for i in changing_indices
            )
        print(line)

    print("-" * len(header))
    print("Note: '.' indicates the token remained unchanged from the previous step.")

def print_sudoku(sudoku, tokenizer):
    for i, row in enumerate(sudoku):
        for j, element in enumerate(row):
            if element == tokenizer.mask_token_id:
                element = '.'
            elif element >= 9:
                element = 'X'
            print(element, end=' ')
            if (j + 1) % 3 == 0:
                print("", end=' ')
        print()
        if (i + 1) % 3 == 0:
            print()

def print_history_interactive(histories, diffusion_mask, tokenizer):
    # histories.shape = (num_samples, num_denoising_steps + 1, seq_len)
    if histories[0, 0, 0] == tokenizer.bos_token_id:
        histories = histories[:, :, 1:]
    if histories[0, 0, -1] == tokenizer.eos_token_id:
        histories = histories[:, :, :-1]
    
    num_samples = histories.shape[0]
    num_denoising_steps = histories.shape[1] - 1
    seq_len = histories.shape[2]
    sudoku_size = int(seq_len ** 0.5)
    sample_id = 0
    diffusion_step = 0

    while True:
        print("\033c", end="")  # Clear screen (for better viewing)
        print(f"Sample {sample_id + 1} of {num_samples}, Step {diffusion_step} of {num_denoising_steps}")
        print_sudoku(histories[sample_id, diffusion_step].reshape(sudoku_size, sudoku_size).numpy(), tokenizer)

        cmd = input("\n[w] next sample, [s] prev sample, [d] next step, [a] prev step, [q] quit: ").strip().lower()
        if cmd == "w":
            if sample_id < num_samples - 1:
                sample_id += 1
        elif cmd == "s":
            if sample_id > 0:
                sample_id -= 1
        elif cmd == "d":
            if diffusion_step < num_denoising_steps:
                diffusion_step += 1
        elif cmd == "a":
            if diffusion_step > 0:
                diffusion_step -= 1
        elif cmd == "q":
            break
        else:
            print("Invalid command.")

# Evaluation for samples starting from puzzles
import hydra
import tqdm
import torch
import numpy as np
import os

from functools import partial
from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler


@hydra.main(config_path="../configs", config_name="generate_from_puzzle", version_base="1.1")
def main(config):
# args pass batch size, ckpt_path, num_denoising_steps and min_p
    def add_bos_eos(examples, cols):
        for col in cols:
            examples[col] = [tokenizer.bos_token + example + tokenizer.eos_token for example in examples[col]]
        return examples
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)

    model, noise_schedule, tokenizer, ckpt_config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    ckpt_config.training.eval_batch_size = config.batch_size
    dtype = parse_dtype(ckpt_config.training.dtype)

    ds_eval = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/{ckpt_config.data.dataset_name}{('_' + ckpt_config.data.dataset_subset) if ckpt_config.data.dataset_subset else ''}/evaluate")
    ds_eval = ds_eval.select(range(16))
    num_samples = ds_eval.num_rows
    
    print(f"Evaluating model from {ckpt_path} on {num_samples} samples")
    
    # ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle', 'solution']), batched=True)
    ds_eval = ds_eval.map(partial(add_bos_eos, cols=['puzzle']), batched=True)
    puzzles = ds_eval.select_columns(['puzzle'])
    solutions = ds_eval.select_columns(['solution'])

    puzzles_tokenized = torch.tensor(tokenizer(puzzles['puzzle'])['input_ids'])
    solutions_tokenized = torch.tensor(tokenizer(solutions['solution'])['input_ids'])
    diffusion_mask = (puzzles_tokenized == tokenizer.mask_token_id).to(int)
    sampler = get_sampler(ckpt_config, model, tokenizer, noise_schedule, sampling_config=config, compile_step=config.compilation.compile_torch, min_p=config.min_p)
    model.eval()

    samples = []
    histories = []
    with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, num_samples, config.batch_size):
                bs = min(config.batch_size, num_samples - i)
                # TODO: how is the max_length in SamplerInstance.model.config.max_seq_len set? Once that is done automatically for sudoku, no need to pass it here
                z_t, history = sampler.generate_from_given(puzzles_tokenized[i:i+bs], diffusion_mask[i:i+bs], config.num_denoising_steps, max_length=ckpt_config.model.max_seq_len, decode=False, show_progress=False, keep_history=True)
                samples.append(z_t)
                histories.append(history)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0)
    histories = torch.cat(histories, dim=0).cpu()
    print(histories.shape)
    print(diffusion_mask.shape)
    post_correction_samples = samples.clone()

    samples = samples.cpu()[:, 1:-1]
    pre_correction_samples_path = os.path.join(ckpt_path, "../../samples/", "evaluation_samples_pre_correction.pt")
    torch.save(samples, hydra.utils.to_absolute_path(pre_correction_samples_path))

    score_samples(samples, diffusion_mask, solutions_tokenized, tokenizer)
    
    # print_history_interactive(histories, diffusion_mask, tokenizer)

    print_sudoku(puzzles_tokenized[0, 1:-1].reshape(9, 9).numpy(), tokenizer)
    print_sudoku(samples[0].reshape(9, 9).numpy(), tokenizer)
    print_sudoku(solutions_tokenized[0].reshape(9, 9).numpy(), tokenizer)

    # print(histories[0, :, :5])
    # print(puzzles_tokenized[0])
    # show_history(histories, diffusion_mask)
    # print(histories[0, :, 8])

    # with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
    #     with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
    #         for i in range(0, num_samples, args.batch_size):
    #             bs = min(args.batch_size, num_samples - i)
    #             # TODO: run self-correction on the samples



if __name__ == "__main__":
    main()