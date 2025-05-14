from datasets import load_dataset

def build_sudoku_dataset():
    def extract_dataset_subset(dataset, set_name):
        return dataset.filter(lambda x: x["set"] == set_name)
    
    def add_diffusion_mask(examples):
        # TODO: add diffusion mask
        examples['diffusion_mask'] = [ for example in examples['puzzle']]
    
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


build_sudoku_dataset()