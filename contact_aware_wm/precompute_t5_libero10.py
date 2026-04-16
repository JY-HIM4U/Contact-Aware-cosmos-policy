#!/usr/bin/env python3
"""Pre-compute T5 embeddings for libero_10 task descriptions and add to cache."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import torch
from huggingface_hub import hf_hub_download
from cosmos_policy._src.predict2.inference.get_t5_emb import get_text_embedding
from libero.libero import benchmark

# Load existing cache
t5_path = hf_hub_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
                           "libero_t5_embeddings.pkl")
with open(t5_path, "rb") as f:
    cache = pickle.load(f)
print(f"Existing cache has {len(cache)} entries:")
for k in cache:
    print(f"  - {k}")

# Get task descriptions for libero_10
tasks = [
    ("libero_10", 2),
    ("libero_10", 3),
    ("libero_10", 9),
]

new_labels = []
for suite_name, task_idx in tasks:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_name = suite.get_task_names()[task_idx]
    task_desc = task_name.replace("_", " ")
    new_labels.append(task_desc)
    print(f"\n{suite_name}[{task_idx}]: {task_desc}")

# Compute missing embeddings
for label in new_labels:
    if label in cache:
        print(f"  [cached] {label}")
    else:
        print(f"  [computing] {label} ...")
        emb = get_text_embedding(label)
        cache[label] = emb.cpu()
        print(f"  [done] shape={emb.shape}")

# Save updated cache
out_path = t5_path  # overwrite the HF cached file
with open(out_path, "wb") as f:
    pickle.dump(cache, f)
print(f"\nSaved {len(cache)} embeddings to {out_path}")

# Free T5 model
del emb
torch.cuda.empty_cache()
print("Done!")
