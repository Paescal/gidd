from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("tokenizers/sudoku")

ds = load_from_disk(f"datasets/sudoku_3m/evaluate")
solutions = ds['solution']
puzzles = ds['puzzle']

# eval/generate_samples.py should work for sampling from sudoku model.
# To start the denoising process from a partially denoised sample, use sampler(instance).sampling_step(), see sampling.py _do_generate()