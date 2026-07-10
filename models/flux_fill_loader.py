"""models.flux_fill_loader — load frozen FLUX.1 Fill [dev] with structure checks.

Responsibilities
  1. Load FluxFillPipeline components in bf16 (plan Sec. 4 memory priority 1).
  2. HARD-ASSERT the structural assumptions the sparse machinery relies on:
       - 19 dual-stream blocks, 38 single-stream blocks
       - transformer.config.in_channels == 384 (64 latent + 320 mask cond)
       - single blocks expose norm / proj_mlp / act_mlp / attn / proj_out
       - attn exposes to_q/to_k/to_v (+ norm_q/norm_k)
     If any assert fails (diffusers refactor), fail loudly with the pinned
     version from requirements.txt instead of silently mis-caching.
  3. Memory strategy helpers: text-encoder-only loading (for prompt_cache),
     VAE tiling/slicing toggles, and unloading text encoders after use.

Everything is frozen: requires_grad_(False) on target components always.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
EXPECTED_DUAL = 19
EXPECTED_SINGLE = 38
EXPECTED_IN_CHANNELS = 384
PINNED_HINT = ("structure mismatch — this code targets the FluxTransformer2DModel "
               "layout of the diffusers version pinned in requirements.txt; "
               "pip install the pinned version or update flux_sparse_transformer.py")


def _assert_structure(transformer):
    assert len(transformer.transformer_blocks) == EXPECTED_DUAL, PINNED_HINT
    assert len(transformer.single_transformer_blocks) == EXPECTED_SINGLE, PINNED_HINT
    assert transformer.config.in_channels == EXPECTED_IN_CHANNELS, PINNED_HINT
    blk = transformer.single_transformer_blocks[0]
    for attr in ("norm", "proj_mlp", "act_mlp", "attn", "proj_out"):
        assert hasattr(blk, attr), f"single block missing .{attr}: {PINNED_HINT}"
    for attr in ("to_q", "to_k", "to_v"):
        assert hasattr(blk.attn, attr), f"single attn missing .{attr}: {PINNED_HINT}"
    assert hasattr(transformer, "norm_out") and hasattr(transformer, "proj_out"), PINNED_HINT
    # Fix 2: the manual single-block path assumes the STOCK attention processor.
    # LoRA / fused-QKV / IP-Adapter / custom processors change the numeric path
    # relative to _single_block_dense — block them at load time, not at Gate B.
    for j, b in enumerate(transformer.single_transformer_blocks):
        attn = b.attn
        proc = attn.processor.__class__.__name__
        assert proc == "FluxAttnProcessor2_0", \
            f"single block {j}: unexpected attn processor {proc} — {PINNED_HINT}"
        assert attn.norm_q is not None and attn.norm_k is not None, \
            f"single block {j}: missing QK RMSNorm — {PINNED_HINT}"
        assert getattr(attn, "fused_projections", False) is False, \
            f"single block {j}: fused QKV active; unfuse before running — {PINNED_HINT}"
        assert getattr(attn, "added_kv_proj_dim", None) is None, \
            f"single block {j}: added KV projections (IP-Adapter?) — {PINNED_HINT}"
    heads = transformer.single_transformer_blocks[0].attn.heads
    assert heads == transformer.config.num_attention_heads, PINNED_HINT
    # Lever A: manual dual path가 가정하는 서브모듈 (Gate B0-dual과 짝)
    db = transformer.transformer_blocks[0]
    for attr in ("norm1", "norm1_context", "attn", "norm2", "norm2_context",
                 "ff", "ff_context"):
        assert hasattr(db, attr), f"dual block missing .{attr}: {PINNED_HINT}"
    for attr in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                 "norm_added_q", "norm_added_k"):
        assert hasattr(db.attn, attr), f"dual attn missing .{attr}: {PINNED_HINT}"


@dataclass
class FluxComponents:
    pipe: object                 # full FluxFillPipeline (official baseline, Stage 0)
    transformer: torch.nn.Module
    vae: torch.nn.Module
    scheduler: object
    device: str
    dtype: torch.dtype


def load_flux_fill(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    vae_tiling: bool = False,
    keep_text_encoders: bool = True,
) -> FluxComponents:
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    _assert_structure(pipe.transformer)

    for m in (pipe.transformer, pipe.vae):
        m.requires_grad_(False).eval()
    if vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    if not keep_text_encoders:
        unload_text_encoders(pipe)

    return FluxComponents(
        pipe=pipe, transformer=pipe.transformer, vae=pipe.vae,
        scheduler=pipe.scheduler, device=device, dtype=dtype,
    )


def unload_text_encoders(pipe):
    """Memory priority 3: drop both text encoders after prompt embeds are cached."""
    for name in ("text_encoder", "text_encoder_2"):
        enc = getattr(pipe, name, None)
        if enc is not None:
            enc.to("cpu")
            setattr(pipe, name, None)
    torch.cuda.empty_cache()


class _TextEncoderOnly:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def encode(self, prompt: str):
        out = self.pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device=self.device)
        prompt_embeds, pooled_prompt_embeds = out[0], out[1]
        return prompt_embeds, pooled_prompt_embeds

    def unload(self):
        unload_text_encoders(self.pipe)


def load_text_encoders_only(model_id: str = MODEL_ID, device: str = "cuda"):
    """Load pipeline WITHOUT the 12B transformer to encode prompts cheaply.
    NOTE: vae must NOT be None — FluxFillPipeline.__init__ reads
    self.vae.config.latent_channels at construction time. The VAE is ~300MB
    and stays idle here; only the transformer is skipped."""
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(
        model_id, transformer=None, torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    return _TextEncoderOnly(pipe, device)
