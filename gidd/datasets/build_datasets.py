from datasets import load_dataset

def build_sudoku():
    def extract_dataset_subset(dataset, set_name):
        return dataset.filter(lambda x: x["set"] == set_name)
    
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split="train")
    subset = "3m"
    evaluation_size = 10_000
    ds = extract_dataset_subset(ds, subset)
    ds = ds.train_test_split(test_size=evaluation_size)
    ds_train = ds['train'].select_columns(['solution'])
    ds_evaluate = ds['test']
    ds_train.save_to_disk(f"gidd/datasets/sudoku_{subset}/train")
    ds_evaluate.save_to_disk(f"gidd/datasets/sudoku_{subset}/evaluate")
    print(ds_train)
    print(ds_evaluate)

build_sudoku()