"""Augment t5_embeddings.pkl with the libero_90 language strings that are missing.

Writes t5_embeddings_with_libero90.pkl next to the original so existing training runs
are not affected.

Usage:
    python contact_aware_wm/augment_t5_libero90.py
"""

from __future__ import annotations

import glob
import os
import pickle
import re

import torch

LIB_BDDL = os.path.expanduser(
    "~/Projects/Contact-Aware-cosmos-policy/.venv/lib/python3.10/site-packages/libero/libero/bddl_files/libero_90"
)
IN_PKL = os.path.expanduser(
    "~/data/LIBERO-Cosmos-Policy/success_only/t5_embeddings.pkl"
)
OUT_PKL = os.path.expanduser(
    "~/data/LIBERO-Cosmos-Policy/success_only/t5_embeddings_with_libero90.pkl"
)


def parse_lang(path: str) -> str | None:
    m = re.search(r"\(:language\s+([^)]+)\)", open(path).read())
    return m.group(1).strip() if m else None


def main() -> None:
    with open(IN_PKL, "rb") as f:
        embs = pickle.load(f)
    print(f"Existing embeddings: {len(embs)}")

    langs = {parse_lang(p) for p in glob.glob(f"{LIB_BDDL}/*.bddl")}
    langs.discard(None)
    missing = sorted([s for s in langs if s not in embs])
    print(f"libero_90 unique language strings: {len(langs)}")
    print(f"missing (need to generate): {len(missing)}")

    if not missing:
        print("nothing to do")
        return

    from transformers import T5EncoderModel, T5TokenizerFast

    print("Loading google-t5/t5-11b encoder in bf16 on cuda:0 ...")
    tok = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
    enc = T5EncoderModel.from_pretrained("google-t5/t5-11b", torch_dtype=torch.bfloat16).to("cuda:0")
    enc.eval()

    @torch.inference_mode()
    def emb_one(s: str) -> torch.Tensor:
        b = tok([s], return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        ids = b.input_ids.to("cuda:0")
        mask = b.attention_mask.to("cuda:0")
        out = enc(input_ids=ids, attention_mask=mask).last_hidden_state  # (1, 512, 1024)
        lengths = mask.sum(dim=1).cpu()
        out[0, lengths[0]:] = 0
        return out.to(dtype=torch.bfloat16).cpu()

    for i, s in enumerate(missing):
        embs[s] = emb_one(s)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(missing)}  '{s[:50]}...'")

    # Sanity check
    k = next(iter(embs))
    print(f"Final count: {len(embs)}, shape[{k[:30]}...]={embs[k].shape}, dtype={embs[k].dtype}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(embs, f)
    print(f"Saved -> {OUT_PKL}")


if __name__ == "__main__":
    main()
