import tqdm
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset


def extract_dataset_subset(dataset, set_name):
        return dataset.filter(lambda x: x["set"] == set_name)
    
def add_diffusion_mask(examples):
    examples['diffusion_mask'] = [''.join('1' if c == '0' else '0' for c in example) for example in examples['puzzle']]
    return examples


def build_sudoku_dataset():
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    subset = "3m"
    evaluation_size = 10_000
    ds = extract_dataset_subset(ds, subset)
    ds = ds.train_test_split(test_size=evaluation_size)
    ds_train = ds['train'].select_columns(['puzzle', 'solution'])
    ds_train = ds_train.map(add_diffusion_mask, batched=True)
    ds_train = ds_train.rename_column('solution', 'text')
    ds_evaluate = ds['test']

    ds_train.save_to_disk(f"gidd/datasets/sudoku_{subset}/train")
    ds_evaluate.save_to_disk(f"gidd/datasets/sudoku_{subset}/evaluate")

    ds_train.to_parquet(f"gidd/datasets/sudoku/train/{subset}.parquet")
    ds_evaluate.to_parquet(f"gidd/datasets/sudoku/evaluate/{subset}.parquet")

    print(ds_train)
    print(ds_evaluate)

    # ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    # subset_names = ["1m", "3m", "4m", "9m", "challenge"]
    # subsets = [extract_dataset_subset(ds, subset) for subset in subset_names]


def parse_shah_dataset():
    def parse_example(example):
        parsed_solution = np.zeros((81), dtype=int)
        parsed_puzzle = np.zeros((81), dtype=int)
        num_given = example[0]
        for i in range(1, 4 * num_given + 1, 4):
            row = example[i]
            col = example[i + 1]
            val = example[i + 2]
            strategy = example[i + 3]
            assert strategy == 0
            parsed_solution[row * 9 + col] = val
            parsed_puzzle[row * 9 + col] = val
        for i in range(4 * num_given + 1, len(example), 4):
            row = example[i]
            col = example[i + 1]
            val = example[i + 2]
            strategy = example[i + 3]
            assert strategy != 0
            parsed_solution[row * 9 + col] = val
            parsed_puzzle[row * 9 + col] = 0
        solution_as_str = "".join([str(x) for x in parsed_solution])
        puzzle_as_str = "".join([str(x) for x in parsed_puzzle])
        return solution_as_str, puzzle_as_str
    
    easy_data_train = np.load("gidd/datasets/sudoku_shah/sudoku-train-data.npy", allow_pickle=False)
    easy_data_test = np.load("gidd/datasets/sudoku_shah/sudoku-test-data.npy", allow_pickle=False)
    easy_data_train = easy_data_train.astype(np.int32)
    easy_data_test = easy_data_test.astype(np.int32)

    easy_data_train_parsed = list(map(parse_example, easy_data_train))
    easy_data_train_solutions, easy_data_train_puzzles = zip(*easy_data_train_parsed)

    easy_data_test_parsed = list(map(parse_example, easy_data_test))
    easy_data_test_solutions, easy_data_test_puzzles = zip(*easy_data_test_parsed)

    ds_easy_train = Dataset.from_dict({'text': easy_data_train_solutions, 'puzzle': easy_data_train_puzzles})
    ds_easy_test = Dataset.from_dict({'text': easy_data_test_solutions, 'puzzle': easy_data_test_puzzles})

    ds_easy_train = ds_easy_train.map(add_diffusion_mask, batched=True)
    ds_easy_test = ds_easy_test.map(add_diffusion_mask, batched=True)

    print(ds_easy_train)
    print(ds_easy_test)

    ds_easy_train.save_to_disk(f"gidd/datasets/sudoku_shah/easy/train")
    ds_easy_test.save_to_disk(f"gidd/datasets/sudoku_shah/easy/test")


def build_sudoku_shah(parse_shah_dataset=True):
    if parse_shah_dataset:
        parse_shah_dataset()

    ds_easy_train = load_from_disk(f"gidd/datasets/sudoku_shah/easy/train")
    ds_easy_test = load_from_disk(f"gidd/datasets/sudoku_shah/easy/test")

    easy_train_set = set(ds_easy_train['text'])
    easy_test_set = set(ds_easy_test['text'])

    easy_all_set = easy_train_set | easy_test_set
    
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    subset = "3m"
    ds = extract_dataset_subset(ds, subset)
    ds = ds.rename_column('solution', 'text')
    print(ds)

    hard_data = []
    for row in tqdm.tqdm(ds, desc="Filtering hard data", total=ds.num_rows):
        if row['text'] not in easy_all_set:
            hard_data.append([row['text'], row['puzzle']])
    hard_data = np.array(hard_data)

    ds_hard = Dataset.from_dict({'text': hard_data[:, 0], 'puzzle': hard_data[:, 1]})
    ds_hard = ds_hard.map(add_diffusion_mask, batched=True)

    print(ds_hard)
    
    ds_hard.save_to_disk(f"gidd/datasets/sudoku_shah/hard/test")


# build_sudoku_dataset()
# parse_shah_dataset()
# build_sudoku_shah(parse_shah_dataset=False)