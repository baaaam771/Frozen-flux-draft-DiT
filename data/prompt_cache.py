"""data.prompt_cache — precompute & cache FLUX prompt embeddings.

Memory strategy step 2–3 of the plan: encode every benchmark prompt once with
the two text encoders (CLIP pooled + T5), save to disk, then run all sampling
with the text encoders unloaded. This removes ~10GB from the sampling-time
footprint and makes every method share byte-identical conditioning.

Usage:
    python -m data.prompt_cache --manifest data/coco_manifest.json \\
        --out /mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def prompt_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:16]


def cache_path(cache_dir: str, prompt: str) -> Path:
    return Path(cache_dir) / f"{prompt_key(prompt)}.pt"


@torch.no_grad()
def build_cache(manifest_json: str, cache_dir: str, model_id: str, device: str = "cuda"):
    from models.flux_fill_loader import load_text_encoders_only

    enc = load_text_encoders_only(model_id, device=device)
    with open(manifest_json) as f:
        items = json.load(f)["items"]
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    seen = set()
    n_new = n_existing = 0
    for it in items:
        p = it["prompt"]
        if p in seen:
            continue
        seen.add(p)
        out = cache_path(cache_dir, p)
        if out.exists():
            n_existing += 1
            continue
        prompt_embeds, pooled = enc.encode(p)
        torch.save(
            {"prompt": p,
             "prompt_embeds": prompt_embeds.to(torch.bfloat16).cpu(),
             "pooled_prompt_embeds": pooled.to(torch.bfloat16).cpu()},
            out,
        )
        n_new += 1
    enc.unload()
    print(f"unique prompts={len(seen)}, newly cached={n_new}, "
          f"already existed={n_existing} -> {cache_dir}")


def load_cached(cache_dir: str, prompt: str, device, dtype=torch.bfloat16):
    d = torch.load(cache_path(cache_dir, prompt), map_location="cpu")
    return (d["prompt_embeds"].to(device, dtype),
            d["pooled_prompt_embeds"].to(device, dtype))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="black-forest-labs/FLUX.1-Fill-dev")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    build_cache(a.manifest, a.out, a.model, a.device)
