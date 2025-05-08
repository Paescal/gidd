from transformers import AutoTokenizer
from datasets import load_from_disk

# eval/generate_samples.py should work for sampling from sudoku model.
# To start the denoising process from a partially denoised sample, use sampler(instance).sampling_step(), see sampling.py _do_generate()


tokenizer = AutoTokenizer.from_pretrained("tokenizers/sudoku_padded")

ds = load_from_disk(f"datasets/sudoku_3m/evaluate")
solutions = ds['solution']
puzzles = ds['puzzle']

# For each row and each column, check whether the numbers 1 to sudoku_size (e.g. 9 for 9x9) are present. If so, add 1 to the score.
# Expects the sudoku to be a 2D array of shape (sudoku_size, sudoku_size).
def row_col_set_score(sudoku):
    row_sets = [set(row) for row in sudoku]
    col_sets = [set(col) for col in sudoku.T]
    score = 0
    for curr_set in row_sets + col_sets:
        for i in range(len(sudoku)):
            if str(i + 1) in curr_set:
                score += 1
    return score