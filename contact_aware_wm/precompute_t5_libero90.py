#!/usr/bin/env python3
"""Precompute T5 embeddings for libero_90 (+ libero_10) NL instructions.

Source of truth for keys: the `language_instruction` stored in each HDF5's
`data.attrs["problem_info"]`. This is what `LIBERODataset` uses as `ep["command"]`
and also what `LIBERODatasetFT` filters against.

Do NOT key by `benchmark.get_task_names()`: those include scene prefixes like
`KITCHEN_SCENE10_...` that the dataset strips. Early v1 of this script made that
mistake and produced a pkl that matched 3/90 tasks.

Starts from the HF-shipped `libero_t5_embeddings.pkl` (40 entries covering
libero_spatial/goal/object + all 10 libero_10) and only encodes what's missing
(~72 prompts for libero_90). The T5-11B model auto-redownloads to HF cache on
first run (~85 G).

Usage:
    conda activate cosmos
    python contact_aware_wm/precompute_t5_libero90.py \
        --out_pkl /home/jaeyoun/contact-aware-wm/data/libero_90_with_ft/t5_embeddings.pkl
"""
import argparse
import glob
import json
import os
import pickle
import sys

import h5py

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf_pkl",
        default="/home/jaeyoun/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B"
                "/snapshots/cb689ec0e3347c13667d70a78a3447388f5c3bb8/libero_t5_embeddings.pkl",
    )
    p.add_argument(
        "--out_pkl",
        default="/home/jaeyoun/contact-aware-wm/data/libero_90_with_ft/t5_embeddings.pkl",
    )
    p.add_argument(
        "--hdf5_dirs",
        nargs="+",
        default=[
            "/home/jaeyoun/LIBERO/datasets/libero_90/libero_90",
            "/home/jaeyoun/LIBERO/datasets/libero_10",
        ],
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def collect_nl_instructions(hdf5_dirs):
    cmds = set()
    for d in hdf5_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.hdf5"))):
            try:
                with h5py.File(f, "r") as h:
                    pi = json.loads(h["data"].attrs["problem_info"])
                    cmds.add(pi["language_instruction"])
            except Exception as e:
                print(f"  skip {os.path.basename(f)}: {e}", file=sys.stderr)
    return sorted(cmds)


def main():
    args = parse_args()

    if os.path.exists(args.hf_pkl):
        with open(args.hf_pkl, "rb") as f:
            cache = pickle.load(f)
        print(f"[precompute] Loaded {len(cache)} entries from HF cache.")
    else:
        cache = {}
        print(f"[precompute] No HF cache at {args.hf_pkl}; starting empty.")

    wanted = collect_nl_instructions(args.hdf5_dirs)
    missing = [c for c in wanted if c not in cache]
    print(f"[precompute] {len(wanted)} unique NL instructions across {args.hdf5_dirs}")
    print(f"[precompute] {len(missing)} missing, will encode.")

    if missing:
        import torch
        from tqdm import tqdm
        from transformers import T5EncoderModel, T5TokenizerFast

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

        print(f"[precompute] Loading T5-11B ({args.dtype}, device_map=auto) — may download 85 G on first run ...")
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
        model = T5EncoderModel.from_pretrained(
            "google-t5/t5-11b",
            dtype=dtype,
            device_map="auto",
        )
        model.eval()
        first_device = next(model.parameters()).device

        with torch.inference_mode():
            for i in tqdm(range(0, len(missing), args.batch_size), desc="batches"):
                batch = missing[i : i + args.batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                )
                input_ids = enc["input_ids"].to(first_device)
                attn = enc["attention_mask"].to(first_device)
                out = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
                lengths = attn.sum(dim=1).cpu()
                for b_idx, prompt in enumerate(batch):
                    row = out[b_idx : b_idx + 1].clone()  # (1, 512, 1024)
                    row[0, lengths[b_idx] :] = 0
                    cache[prompt] = row.to(torch.bfloat16).cpu()

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(cache, f)
    print(f"[precompute] Wrote {len(cache)} entries to {args.out_pkl}")

    # Coverage report
    hit_90 = sum(
        1 for c in wanted
        if c in cache and any("libero_90" in d for d in args.hdf5_dirs)
    )
    print(f"[precompute] Hit {len(set(wanted) & set(cache))} / {len(wanted)} wanted instructions.")


if __name__ == "__main__":
    main()
