import functools
import hashlib
import json
import os
import shutil
from typing import Callable

import numpy as np
import omegaconf
import torch
import hydra
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataset(config, num_proc=32):
    test_size = int(config.data.test_size)
    n_proc = min(os.cpu_count(), num_proc)
    if config.data.local_dataset:
        # TODO?: run dataloading script to download and prepare the dataset if it does not exist
        # expects sudoku.yaml to contain e.g. dataset_name: sudoku, dataset_subset: 3m
        # ds = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/{config.data.dataset_name}{('_' + config.data.dataset_subset) if config.data.dataset_subset else ''}/train")
        # ds = ds.train_test_split(test_size=test_size)
        # train_ds = ds['train']
        # test_ds = ds['test']
        # score_ds = test_ds

        ds = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/easy/train")
        ds = ds.train_test_split(test_size=test_size)
        train_ds = ds['train']
        print(f"train dataset size: {len(train_ds)}")
        test_ds = ds['test']
        score_ds = load_from_disk(f"/local/home/prisold/gidd/gidd/datasets/sudoku_shah/hard/test")
        # score_ds = test_ds
        
        # expects sudoku.yaml to contain e.g. dataset_name: gidd/datasets/sudoku/train/, dataset_subset: 3m
        # train_ds = load_dataset(
        #     config.data.dataset_name,
        #     config.data.dataset_subset,
        #     split=f"train[:-{test_size}]",
        #     trust_remote_code=config.data.trust_remote_code,
        #     num_proc=n_proc,
        # )
        # test_ds = load_dataset(
        #     config.data.dataset_name,
        #     config.data.dataset_subset,
        #     split=f"train[-{test_size}:]",
        #     trust_remote_code=config.data.trust_remote_code,
        #     num_proc=n_proc,
        # )
    else:
        train_ds = load_dataset(
            config.data.dataset_name,
            config.data.dataset_subset,
            split=f"train[:-{test_size}]",
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
        )
        test_ds = load_dataset(
            config.data.dataset_name,
            config.data.dataset_subset,
            split=f"train[-{test_size}:]",
            trust_remote_code=config.data.trust_remote_code,
            num_proc=n_proc,
        )

    return train_ds, test_ds, score_ds


def cached_dataset(cache_dir: str, file_name: str, generate_fn: Callable[[], Dataset]) -> Dataset:
    if cache_dir is None:
        return generate_fn()

    cache_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_path):
        ds = Dataset.load_from_disk(cache_path)
        return ds
    else:
        ds = generate_fn()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            ds.save_to_disk(cache_path)
        except Exception as e:
            shutil.rmtree(cache_path)
            raise e
        return ds


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 512,
    sequence_packing: bool = False,
    batch_size: int = 1024,
    num_proc: int = 32,
):
    n_proc = min(os.cpu_count(), num_proc)
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    tokenizer_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = 10_000_000

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
        )["input_ids"]
        tokens = [[bos_token_id] + x + ([] if sequence_packing else [eos_token_id]) for x in tokens]
        if sequence_packing:
            tokens = np.concatenate(tokens, axis=0)
            tokens = tokens[: len(tokens) - len(tokens) % max_seq_len]
            tokens = tokens.reshape(-1, max_seq_len)
        else:
            tokens = [
                np.pad(x, (0, max_seq_len - len(x) % max_seq_len), mode="constant", constant_values=tokenizer.pad_token_id)
                for x in tokens
            ]
            tokens = [x.reshape(-1, max_seq_len) for x in tokens]
            tokens = np.concatenate(tokens, axis=0)
        return {"input_ids": tokens}

    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=n_proc,
    )

    tokenizer.model_max_length = tokenizer_max_len
    return ds


def default_collator(config, tokenizer, examples, text_key="text"):
    puzzle_seq_len = config.model.max_seq_len
    if config.training.use_diffusion_mask:
        diffusion_masks = [x['diffusion_mask'] for x in examples]
    else:
        diffusion_masks = ['1' * puzzle_seq_len for x in examples]
    solutions = [x[text_key] for x in examples]
    puzzles = [x['puzzle'] for x in examples]
    solution_tokens = tokenizer(solutions, truncation=False, return_tensors="np")
    puzzle_tokens = tokenizer(puzzles, truncation=False, return_tensors="np")

    try:
        use_in_context = config.model.puzzle_conditioning == 'in_context'
    except omegaconf.errors.ConfigAttributeError:
        use_in_context = False
    if use_in_context:
        solution_ids = np.concatenate((puzzle_tokens["input_ids"], solution_tokens["input_ids"]), axis=-1)
        puzzle_ids = np.concatenate((puzzle_tokens["input_ids"], puzzle_tokens["input_ids"]), axis=-1)
        diffusion_masks = [[0 for _ in range(puzzle_seq_len)] + [int(c) for c in list(mask)] for mask in diffusion_masks]
        attention_masks = np.concatenate((np.ones_like(puzzle_tokens["attention_mask"]), solution_tokens["attention_mask"]), axis=-1)
    else:
        solution_ids = solution_tokens["input_ids"]
        puzzle_ids = puzzle_tokens["input_ids"]
        diffusion_masks = [[int(c) for c in list(mask)] for mask in diffusion_masks]
        attention_masks = solution_tokens["attention_mask"]
    
    solution_ids = torch.from_numpy(np.array(solution_ids)).to(torch.long)
    puzzle_ids = torch.from_numpy(np.array(puzzle_ids)).to(torch.long)
    diffusion_masks = torch.from_numpy(np.array(diffusion_masks)).to(torch.long)
    attention_masks = torch.from_numpy(np.array(attention_masks)).to(torch.int)
    try:
        use_in_context = config.model.puzzle_conditioning == 'in_context'
    except omegaconf.errors.ConfigAttributeError:
        use_in_context = False
    if use_in_context:
        max_length = config.model.max_seq_len * 2
    else:
        max_length = config.model.max_seq_len
    assert solution_ids.shape[1] == max_length
    assert puzzle_ids.shape[1] == max_length
    assert diffusion_masks.shape[1] == max_length
    assert attention_masks.shape[1] == max_length
    return BatchEncoding({"input_ids": solution_ids, "puzzle_ids": puzzle_ids, "diffusion_mask": diffusion_masks, "attention_mask": attention_masks}, tensor_type="pt", n_sequences=len(solution_ids))


def pretokenized_collator(examples, pad_token_id=0, tokens_key="input_ids"):
    input_ids = np.stack([np.array(x[tokens_key]) for x in examples], axis=0)
    attn_masks = (input_ids != pad_token_id).astype(np.int32)
    input_ids = torch.from_numpy(input_ids).to(torch.long)
    attn_masks = torch.from_numpy(attn_masks).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def subsample_collator(config, tokenizer, examples, text_key="text"):
    # bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    # eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    puzzles = [x['puzzle'] for x in examples]
    if config.training.use_diffusion_mask:
        diffusion_masks = [x['diffusion_mask'] for x in examples]
    else:
        diffusion_masks = ['1' * len(x['diffusion_mask']) for x in examples]
    solutions = [x[text_key] for x in examples]
    solution_tokens = tokenizer(solutions, truncation=False, return_tensors="np")
    puzzle_tokens = tokenizer(puzzles, truncation=False, return_tensors="np")
    max_length = config.model.max_seq_len
    solution_ids = []
    puzzle_ids = []
    diffusion_masks_padded = []
    attn_masks = []
    for i in range(len(solutions)):
        solution_toks = solution_tokens["input_ids"][i]
        puzzle_toks = puzzle_tokens["input_ids"][i]
        diffusion_mask = [int(c) for c in list(diffusion_masks[i])]
        attn_mask = solution_tokens["attention_mask"][i]
        # if solution_toks[0] != bos_token_id:
        #     solution_toks = np.concatenate([[bos_token_id], solution_toks])
        #     puzzle_toks = np.concatenate([[bos_token_id], puzzle_toks])
        #     diffusion_mask = np.concatenate(([0], diffusion_mask))
        #     attn_mask = np.concatenate([[1], attn_mask])
        # if solution_toks[-1] != eos_token_id:
        #     solution_toks = np.concatenate([solution_toks, [eos_token_id]])
        #     puzzle_toks = np.concatenate([puzzle_toks, [eos_token_id]])
        #     diffusion_mask = np.concatenate([diffusion_mask, [0]])
        #     attn_mask = np.concatenate([attn_mask, [1]])

        if len(solution_toks) > max_length:
            overflow = len(solution_toks) - max_length
            start_idx = np.random.randint(0, overflow + config.data.max_add_padding)
            solution_toks = solution_toks[start_idx : start_idx + max_length]
            puzzle_toks = puzzle_toks[start_idx : start_idx + max_length]
            diffusion_mask = diffusion_mask[start_idx : start_idx + max_length]
            attn_mask = attn_mask[start_idx : start_idx + max_length]
        if len(solution_toks) < max_length:
            underflow = max_length - len(solution_toks)
            solution_toks = np.pad(solution_toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            puzzle_toks = np.pad(puzzle_toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            diffusion_mask = np.pad(diffusion_mask, (0, underflow), mode="constant", constant_values=0)
            attn_mask = np.pad(attn_mask, (0, underflow), mode="constant", constant_values=0)
        assert len(solution_toks) == max_length
        assert len(puzzle_toks) == max_length
        assert len(diffusion_mask) == max_length
        assert len(attn_mask) == max_length
        solution_ids.append(solution_toks)
        puzzle_ids.append(puzzle_toks)
        diffusion_masks_padded.append(diffusion_mask)
        attn_masks.append(attn_mask)
    solution_ids = torch.from_numpy(np.array(solution_ids)).to(torch.long)
    puzzle_ids = torch.from_numpy(np.array(puzzle_ids)).to(torch.long)
    diffusion_masks_padded = torch.from_numpy(np.array(diffusion_masks_padded)).to(torch.long)
    attn_masks = torch.from_numpy(np.array(attn_masks)).to(torch.int)
    return BatchEncoding({"input_ids": solution_ids, "puzzle_ids": puzzle_ids, "diffusion_mask": diffusion_masks_padded, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(solution_ids))


def _get_dataloader_with_seed(seed, config, ds, shuffle, drop_last, batch_size, collate_fn, persistent_workers=True):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(ds, seed=seed, shuffle=shuffle)
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def _get_dataloader(config, ds, shuffle, drop_last, batch_size, collate_fn, persistent_workers=True):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(ds, seed=config.training.seed, shuffle=shuffle)
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def get_dataloaders(config, tokenizer, train_batch_size=None, eval_batch_size=None, score_batch_size=None):
    if train_batch_size is None:
        train_batch_size = config.training.train_batch_size
    if eval_batch_size is None:
        eval_batch_size = config.training.eval_batch_size
    if score_batch_size is None:
        score_batch_size = config.training.score_batch_size

    train_ds, test_ds, score_ds = get_dataset(config)

    if config.data.pre_tokenize:
        max_seq_len = config.model.max_seq_len
        sequence_packing = config.data.sequence_packing
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": config.data.dataset_name,
                    "subset": config.data.dataset_subset,
                    "tokenizer_name": config.data.tokenizer_name,
                    "max_seq_len": max_seq_len,
                    "sequence_packing": sequence_packing,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        train_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-train-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=train_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )
        test_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-test-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=test_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )
        score_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-score-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=score_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )

        collate_fn = functools.partial(pretokenized_collator, pad_token_id=tokenizer.pad_token_id, tokens_key="input_ids")
    else:
        if config.data.sequence_packing:
            raise ValueError("Sequence packing requires pre-tokenization.")

        # collate_fn = functools.partial(subsample_collator, config, tokenizer, text_key="text")
        collate_fn = functools.partial(default_collator, config, tokenizer, text_key="text") # TODO: can set the text_key here (e.g. to "solution")

    train_dl = _get_dataloader(config, train_ds, shuffle=True, drop_last=True, batch_size=train_batch_size, collate_fn=collate_fn)
    test_dl = _get_dataloader(config, test_ds, shuffle=False, drop_last=False, batch_size=eval_batch_size, collate_fn=collate_fn)
    score_dl = _get_dataloader(config, score_ds, shuffle=False, drop_last=False, batch_size=score_batch_size, collate_fn=collate_fn)

    return train_dl, test_dl, score_dl