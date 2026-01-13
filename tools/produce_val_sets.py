"""
Script to produce additional validation sets for NanoScaling experiments.
It saves .bin files to data/openwebtext/ which will be picked up by the trainer.
"""
import os
import argparse
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

def process_dataset_to_bin(dataset_name, split_name, output_name, num_samples=None):
    """
    Downloads `split_name` of `dataset_name`, optionally taking first `num_samples`.
    Tokenizes and saves to data/openwebtext/val_{output_name}.bin
    """
    enc = tiktoken.get_encoding("gpt2")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'openwebtext')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {dataset_name} ({split_name}) -> {output_name}...")
    
    try:
        if dataset_name == 'wikitext':
            ds = load_dataset("wikitext", "wikitext-103-v1", split=split_name)
        elif dataset_name == 'lambada':
            ds = load_dataset("lambada", split=split_name)
        else:
            ds = load_dataset(dataset_name, split=split_name, streaming=True)
            # Materialize if streaming for easier processing if small
            ds = ds.take(num_samples) if num_samples else ds
            ds = Dataset.from_list(list(ds))
            
        if num_samples and not isinstance(ds, type(load_dataset(dataset_name, split=split_name, streaming=True))):
             # if normal dataset
             ds = ds.select(range(min(len(ds), num_samples)))
             
        def process(example):
            text = example['text']
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}

        tokenized = ds.map(process, remove_columns=ds.column_names, desc=f"Tokenizing {output_name}")

        arr_len = np.sum(tokenized['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'val_{output_name}.bin')
        dtype = np.uint16
        
        # Check size fits in memory, otherwise memmap
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        idx = 0
        # Iterate and write
        for item in tqdm(tokenized, desc=f"Writing {filename}"):
            arr_batch = np.array(item['ids'], dtype=dtype)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} ({arr_len} tokens)")

    except Exception as e:
        print(f"Failed to process {dataset_name}: {e}")

if __name__ == "__main__":
    # Create 3 sets
    # 1. Wikitext-103 validation
    process_dataset_to_bin('wikitext', 'validation', 'wikitext', num_samples=None)
    
    # 2. Lambada (Test set often used for long range) - usually 'validation' or 'test'. 
    # Lambada on HF has 'test' and 'validation'.
    # We'll take a subset of validation for speed.
    process_dataset_to_bin('lambada', 'validation', 'lambada', num_samples=1000)
    
    # 3. PTB (Penn Treebank) or just a different domain like code?
    # Let's try a bit of code. 'codeparrot/codeparrot-train-v2-near-dedup' is huge, maybe 'openai_humaneval' (very small)
    # or just wikitext test split as 'wikitext_test'.
    # Let's do Wikitext Test split as another one.
    process_dataset_to_bin('wikitext', 'test', 'wikitext_test', num_samples=None)
