"""data.dataset — COCO-val inpainting benchmark: (image, mask, prompt, sample_id).

Plan Sec. 5: each sample must carry image / mask / prompt / sample_id, and the
same image·mask·prompt·latent-seed must be shared by all methods. We freeze
this by writing a manifest JSON once (`build_manifest`) and having every run
read the manifest — never re-sampling masks or prompts at run time.

Layout expected (already used for FreqSpec COCO runs):
    root/
      val2017/*.jpg  (또는 images/val2017/*.jpg — 자동 감지)
      annotations/captions_val2017.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image

from .masks import MaskSpec, make_mask, spec_for_index


def _load_captions(ann_path: Path) -> dict[str, str]:
    with open(ann_path) as f:
        ann = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    caps: dict[str, str] = {}
    for c in ann["annotations"]:
        fn = id2file.get(c["image_id"])
        if fn is not None and fn not in caps:      # first caption per image
            caps[fn] = c["caption"].strip()
    return caps


def build_manifest(
    root: str,
    out_json: str,
    n: int = 500,
    resolution: int = 512,
    mask_seed: int = 0,
    shuffle_seed: int = 1234,
):
    """Write a frozen benchmark manifest: n samples, balanced mask types/buckets."""
    root_p = Path(root)
    caps = _load_captions(root_p / "annotations" / "captions_val2017.json")
    # image dir layout varies: <root>/val2017 (공식 zip 압축 해제 그대로)
    # 또는 <root>/images/val2017 — 존재하는 쪽을 자동 선택
    img_dir = root_p / "val2017"
    if not img_dir.is_dir():
        img_dir = root_p / "images" / "val2017"
    assert img_dir.is_dir(), f"val2017 images not found under {root}"
    files = sorted(caps)
    random.Random(shuffle_seed).shuffle(files)
    files = files[:n]
    items = []
    for i, fn in enumerate(files):
        spec = spec_for_index(sample_id=fn, index=i, seed=mask_seed)
        items.append({
            "sample_id": fn,
            "image": str(img_dir / fn),
            "prompt": caps[fn],
            "mask_type": spec.mask_type,
            "bucket": spec.bucket,
            "mask_seed": spec.seed,
            "latent_seed": 10_000 + i,          # shared by every method
            "resolution": resolution,
        })
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"resolution": resolution, "items": items}, f, indent=1)
    return out_json


def build_manifest_imagedir(
    image_dir: str,
    out_path: str,
    prompt: str,
    n: int = 200,
    resolution: int = 1024,
    mask_seed: int = 0,
    shuffle_seed: int = 1234,
    exts=(".png", ".jpg", ".jpeg"),
):
    """Caption 없는 데이터셋(FFHQ 등)용 frozen manifest — 고정 prompt 사용.
    COCO manifest와 동일 스키마이므로 전체 파이프라인이 그대로 동작."""
    d = Path(image_dir)
    files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in exts)
    assert files, f"no images under {image_dir}"
    random.Random(shuffle_seed).shuffle(files)
    files = files[:n]
    items = []
    for i, fn in enumerate(files):
        spec = spec_for_index(sample_id=fn, index=i, seed=mask_seed)
        items.append({
            "sample_id": fn,
            "image": str(d / fn),
            "prompt": prompt,
            "mask_type": spec.mask_type,
            "bucket": spec.bucket,
            "mask_seed": spec.seed,
            "latent_seed": 10000 + i,
        })
    manifest = {"resolution": resolution, "items": items}
    Path(out_path).write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(items)} items -> {out_path}")
    return manifest


class FluxFillBenchmark(torch.utils.data.Dataset):
    """Reads the frozen manifest; returns tensors ready for the sampler."""

    def __init__(self, manifest_json: str):
        with open(manifest_json) as f:
            m = json.load(f)
        self.items = m["items"]
        self.resolution = m["resolution"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        R = it.get("resolution", self.resolution)
        img = Image.open(it["image"]).convert("RGB").resize((R, R), Image.LANCZOS)
        image = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        spec = MaskSpec(it["sample_id"], it["mask_type"], it["bucket"], it["mask_seed"])
        mask = make_mask(R, R, spec)                                  # [1, R, R]
        return {
            "image": image,                                           # [3, R, R] in [0,1]
            "mask": mask,
            "prompt": it["prompt"],
            "sample_id": it["sample_id"],
            "latent_seed": it["latent_seed"],
            "bucket": it["bucket"],
            "mask_type": it["mask_type"],
        }
