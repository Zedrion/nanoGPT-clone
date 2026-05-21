import os
import requests
import tiktoken
import numpy as np
from datasets import load_dataset

# load the train split of the hellaswag dataset
print("Loading Rowan/hellaswag train split")
data = load_dataset("Rowan/hellaswag", split="train")

# Only take the first 1/10 of the dataset
print("Selecting all rows")
n = len(data)
train_dataset = data.select(range(int(n*0.9 / 10)))
val_dataset = data.select(range(int(n*0.9 / 10), int(n*0.9 / 10) + int(n*0.1 / 10)))

# convert to text: context + all endings + correct ending
print("Converting to text")
train_data = ""
for row in train_dataset:
    endings = row["endings"]
    correct = int(row["label"])
    train_data += (
        f"{row['ctx']}\n"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n"
        f"Which ending makes the most sense?\nAnswer:{[' A', ' B', ' C', ' D'][correct]}\n\n"
    )
    
val_data = ""
for row in val_dataset:
    endings = row["endings"]
    correct = int(row["label"])
    val_data += (
        f"{row['ctx']}\n"
        f"A) {endings[0]}\n"
        f"B) {endings[1]}\n"
        f"C) {endings[2]}\n"
        f"D) {endings[3]}\n"
        f"Which ending makes the most sense?\nAnswer:{[' A', ' B', ' C', ' D'][correct]}\n\n"
    )

# encode with tiktoken gpt2 bpe
print("Encoding with tiktoken gpt2 bpe")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
print("Exporting to bin files")
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 383,545 tokens
# val has 42,698 tokens