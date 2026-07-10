"""models.flux_cache — depth-aligned anchor cache for FLUX Fill (DACE on FLUX).

At each dense anchor step the sparse forward records, for every single-stream
block j, the *input hidden states of the image tokens* (depth-correct context),
plus the final packed prediction v_a. At sparse steps, easy image tokens
contribute to block-j attention through the anchor's block-j state — never a
frozen depth-m state — which is exactly the repair DACE showed is required
(frozen context FID 154.8 -> DACE 61.8 on ImageNet-64 DiT).

Memory (plan Sec. 9 first version): image tokens only, single-stream blocks
only, batch 1. At 512x512: N=1024 tokens x 3072 dim x 38 blocks x bf16
≈ 0.24 GB — comfortably inside RTX Pro 6000 96GB even at 1024² (≈0.96 GB).

`prev_final_prediction` keeps the previous anchor's v for the delta selector
(anchor-to-anchor change), which is free because both anchors ran anyway.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FluxAnchorCache:
    timestep: torch.Tensor | None = None
    step_index: int = -1
    # single_block_inputs[j]: [B, N_img, D] image-token input states of single block j
    single_block_inputs: list[torch.Tensor] = field(default_factory=list)
    # states entering the single-stream stack (== output of dual stream), image part
    entry_image_states: torch.Tensor | None = None
    entry_text_states: torch.Tensor | None = None
    final_prediction: torch.Tensor | None = None       # packed v_a  [B, N_img, 64]
    prev_final_prediction: torch.Tensor | None = None  # previous anchor's v (delta selector)
    image_token_positions: torch.Tensor | None = None  # abs positions in joint seq
    mask_token_map: torch.Tensor | None = None         # [B, N_img] mask coverage
    # anchor-side quantities for the TRUE anchor clean estimate (Fix 1):
    #   x0_hat_a = z_a - sigma_a * v_a   (z_t with stale v_a is NOT the anchor x0)
    anchor_latents: torch.Tensor | None = None          # packed z_a [B, N_img, 64]
    anchor_sigma: torch.Tensor | None = None            # scalar sigma_a
    anchor_clean_estimate: torch.Tensor | None = None   # packed x0_hat_a

    def set_anchor_context(self, latents: torch.Tensor, sigma: torch.Tensor):
        """Call at every anchor step AFTER finish_anchor with the latents/sigma
        the anchor forward consumed. Precomputes x0_hat_a once per anchor."""
        assert self.final_prediction is not None, "set_anchor_context before finish_anchor"
        self.anchor_latents = latents.detach().clone()
        self.anchor_sigma = torch.as_tensor(sigma).detach().to(latents.device).clone()
        self.anchor_clean_estimate = (
            self.anchor_latents
            - self.anchor_sigma.to(self.anchor_latents.dtype) * self.final_prediction
        ).detach()

    def is_empty(self) -> bool:
        return self.final_prediction is None

    def age(self, current_step_index: int) -> int:
        """Cache age in scheduler steps — all DACE error is temporal, bounded by this."""
        return current_step_index - self.step_index

    def begin_anchor(self, timestep: torch.Tensor, step_index: int):
        self.prev_final_prediction = self.final_prediction
        self.timestep = timestep
        self.step_index = step_index
        self.single_block_inputs = []
        self.single_block_kv = []

    def record_single_kv(self, k_img: torch.Tensor, v_img: torch.Tensor):
        self.single_block_kv.append((k_img.detach().contiguous(),
                                     v_img.detach().contiguous()))

    def record_single_input(self, image_states: torch.Tensor):
        # detach + contiguous: cache must never keep autograd graph or views alive
        self.single_block_inputs.append(image_states.detach().contiguous())

    def finish_anchor(self, final_prediction: torch.Tensor,
                      entry_text: torch.Tensor, entry_image: torch.Tensor):
        self.final_prediction = final_prediction.detach()
        self.entry_text_states = entry_text.detach()
        self.entry_image_states = entry_image.detach()

    def vram_bytes(self) -> int:
        n = 0
        for t in self.single_block_inputs:
            n += t.numel() * t.element_size()
        for kv in self.single_block_kv:
            n += kv[0].numel() * kv[0].element_size()
            n += kv[1].numel() * kv[1].element_size()
        for t in (self.entry_image_states, self.entry_text_states, self.final_prediction,
                  self.anchor_latents, self.anchor_clean_estimate):
            if t is not None:
                n += t.numel() * t.element_size()
        return n
