"""models.sd3_sparse_runner — SD3-계열(all-dual MMDiT) 수동 실행기 (v2).

P0 수정 반영:
  * 텍스트/컨텍스트 스트림은 모든 sparse 변형에서 FULLY FRESH (Lever A 정의).
  * K/V 캐시는 이미지 K/V만 저장; sparse step에서 fresh 텍스트 K/V와 결합하고
    hard 이미지 행은 fresh 값을 scatter.
  * easy 이미지 행은 depth별 anchor 상태(img_states[bi])를 사용 — Lever A는
    거기서 K/V를 재계산(2N), Lever A+B는 캐시 재사용(+hard 2k).
  * 'naive'는 runner에 없음: 논문의 보수 정책상 all-dual 모델의 naive는
    dense와 동일 (sparse-eligible single 블록 부재, floor 1.0).

변형: dense / dual(Lever A) / dualkv(Lever A+B).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _attn(q, k, v, heads):
    B, S, D = q.shape
    hd = D // heads
    q, k, v = (t.view(B, -1, heads, hd).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).reshape(B, -1, D)


def _scatter_rows(dst, idx, src):
    return dst.scatter(1, idx.unsqueeze(-1).expand(-1, -1, dst.shape[-1]), src)


def _gather_rows(t, idx):
    return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))


@dataclass
class SD3AnchorCache:
    img_states: list = field(default_factory=list)   # 블록별 이미지 hidden
    img_kv: list = field(default_factory=list)       # 블록별 joint (k, v)
    self_img_kv: list = field(default_factory=list)  # attn2용 (k, v) | None

    def clear(self):
        self.img_states.clear(); self.img_kv.clear(); self.self_img_kv.clear()

    def vram_bytes(self) -> int:
        n = 0
        for t in self.img_states:
            n += t.numel() * t.element_size()
        for pair in list(self.img_kv) + [p for p in self.self_img_kv
                                         if p is not None]:
            k, v = pair
            n += k.numel() * k.element_size() + v.numel() * v.element_size()
        return n


def _norm1(block, img, temb):
    """use_dual_attention이면 (n_img, g_msa, s, sc, g_mlp, n_img2, g_msa2),
    아니면 뒤 둘이 None."""
    out = block.norm1(img, emb=temb)
    if getattr(block, "use_dual_attention", False):
        return out                                  # 7-tuple
    return (*out, None, None)


class SD3SparseRunner:
    def __init__(self, transformer):
        self.t = transformer
        self.heads = transformer.config.num_attention_heads

    def embed(self, hidden, ctx, timestep, pooled):
        t = self.t
        temb = t.time_text_embed(timestep, pooled)
        return t.pos_embed(hidden), t.context_embedder(ctx), temb

    # -------------------------------------------------- 공통 하위 연산 ----
    def _img_kv(self, attn, n_img):
        k = attn.to_k(n_img)
        v = attn.to_v(n_img)
        k = attn.norm_k(k.view(*k.shape[:2], self.heads, -1)).view_as(k)
        return k, v

    def _attn2_q(self, attn2, n_rows):
        q = attn2.to_q(n_rows)
        if getattr(attn2, "norm_q", None) is not None:
            q = attn2.norm_q(q.view(*q.shape[:2], self.heads, -1)).view_as(q)
        return q

    def _attn2_kv(self, attn2, n_img2):
        k = attn2.to_k(n_img2)
        v = attn2.to_v(n_img2)
        if getattr(attn2, "norm_k", None) is not None:
            k = attn2.norm_k(k.view(*k.shape[:2], self.heads, -1)).view_as(k)
        return k, v

    def _txt_qkv(self, block, n_ctx, pre_only):
        attn = block.attn
        k = attn.add_k_proj(n_ctx)
        v = attn.add_v_proj(n_ctx)
        k = attn.norm_added_k(k.view(*k.shape[:2], self.heads, -1)).view_as(k)
        q = None
        if not pre_only:
            q = attn.add_q_proj(n_ctx)
            q = attn.norm_added_q(
                q.view(*q.shape[:2], self.heads, -1)).view_as(q)
        return q, k, v

    def _ctx_update(self, block, ctx, o_c, c_gates):
        c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = c_gates
        o_c = block.attn.to_add_out(o_c)
        ctx = ctx + c_gate_msa.unsqueeze(1) * o_c
        n2c = block.norm2_context(ctx) * (1 + c_scale_mlp[:, None]) \
            + c_shift_mlp[:, None]
        return ctx + c_gate_mlp.unsqueeze(1) * block.ff_context(n2c)

    # ------------------------------------------------------------ dense ----
    @torch.no_grad()
    def dense_forward(self, hidden, ctx_in, timestep, pooled,
                      record: SD3AnchorCache | None = None):
        img, ctx, temb = self.embed(hidden, ctx_in, timestep, pooled)
        if record is not None:
            record.clear()
        for block in self.t.transformer_blocks:
            pre_only = block.context_pre_only
            n_img, g_msa, s_mlp, sc_mlp, g_mlp, n_img2, g_msa2 = \
                _norm1(block, img, temb)
            if pre_only:
                n_ctx = block.norm1_context(ctx, temb)
                c_gates = None
            else:
                n_ctx, *c_gates = block.norm1_context(ctx, emb=temb)
            attn = block.attn

            if record is not None:
                record.img_states.append(img)
            q_i = attn.to_q(n_img)
            q_i = attn.norm_q(
                q_i.view(*q_i.shape[:2], self.heads, -1)).view_as(q_i)
            k_i, v_i = self._img_kv(attn, n_img)
            if record is not None:
                record.img_kv.append((k_i, v_i))
                record.self_img_kv.append(None)
            q_c, k_c, v_c = self._txt_qkv(block, n_ctx, pre_only)

            T = ctx.shape[1]
            k_full = torch.cat([k_c, k_i], dim=1)
            v_full = torch.cat([v_c, v_i], dim=1)
            q = torch.cat([q_c, q_i], dim=1) if q_c is not None else q_i
            o = _attn(q, k_full, v_full, self.heads)
            if q_c is not None:
                o_c, o_i = o[:, :T], o[:, T:]
            else:
                o_c, o_i = None, o

            img = img + g_msa.unsqueeze(1) * attn.to_out[0](o_i)
            if n_img2 is not None:                    # SD3.5 dual attention
                a2 = block.attn2
                k2, v2 = self._attn2_kv(a2, n_img2)
                if record is not None:
                    record.self_img_kv[-1] = (k2, v2)
                q2 = self._attn2_q(a2, n_img2)
                o2 = a2.to_out[0](_attn(q2, k2, v2, self.heads))
                img = img + g_msa2.unsqueeze(1) * o2
            n2 = block.norm2(img) * (1 + sc_mlp[:, None]) + s_mlp[:, None]
            img = img + g_mlp.unsqueeze(1) * block.ff(n2)
            if o_c is not None and not pre_only:
                ctx = self._ctx_update(block, ctx, o_c, c_gates)
        if record is not None:
            record.img_states.append(img)      # 최종 출력 상태 (final easy 행)
        out = self.t.norm_out(img, temb)
        return self.t.proj_out(out)

    # ----------------------------------------------------------- sparse ----
    @torch.no_grad()
    def sparse_forward(self, hidden, ctx_in, timestep, pooled,
                       cache: SD3AnchorCache, hard: torch.Tensor, lever: str):
        """dual(Lever A) / dualkv(Lever A+B). 텍스트는 fully fresh."""
        assert lever in ("dual", "dualkv")
        img_e, ctx, temb = self.embed(hidden, ctx_in, timestep, pooled)
        img_hard = _gather_rows(img_e, hard)          # hard 행의 진화 상태

        for bi, block in enumerate(self.t.transformer_blocks):
            pre_only = block.context_pre_only
            # easy 행 = depth별 anchor 상태, hard 행 = 현재 진화 상태
            img_full = _scatter_rows(cache.img_states[bi].clone(), hard,
                                     img_hard)
            n_full, g_msa, s_mlp, sc_mlp, g_mlp, n_full2, g_msa2 = \
                _norm1(block, img_full, temb)
            if pre_only:
                n_ctx = block.norm1_context(ctx, temb)
                c_gates = None
            else:
                n_ctx, *c_gates = block.norm1_context(ctx, emb=temb)
            attn = block.attn

            n_hard = _gather_rows(n_full, hard)
            q_i = attn.to_q(n_hard)
            q_i = attn.norm_q(
                q_i.view(*q_i.shape[:2], self.heads, -1)).view_as(q_i)
            if lever == "dual":
                k_img, v_img = self._img_kv(attn, n_full)      # 2N 재계산
            else:
                k_h, v_h = self._img_kv(attn, n_hard)          # 2k fresh
                k_img = _scatter_rows(cache.img_kv[bi][0].clone(), hard, k_h)
                v_img = _scatter_rows(cache.img_kv[bi][1].clone(), hard, v_h)
            q_c, k_c, v_c = self._txt_qkv(block, n_ctx, pre_only)  # text fresh

            T = ctx.shape[1]
            k_full = torch.cat([k_c, k_img], dim=1)
            v_full = torch.cat([v_c, v_img], dim=1)
            q = torch.cat([q_c, q_i], dim=1) if q_c is not None else q_i
            o = _attn(q, k_full, v_full, self.heads)
            if q_c is not None:
                o_c, o_i = o[:, :T], o[:, T:]
            else:
                o_c, o_i = None, o

            img_hard = img_hard + g_msa.unsqueeze(1) * attn.to_out[0](o_i)
            if n_full2 is not None:                   # SD3.5 dual attention
                a2 = block.attn2
                n2_hard = _gather_rows(n_full2, hard)
                q2 = self._attn2_q(a2, n2_hard)
                if lever == "dual":
                    k2, v2 = self._attn2_kv(a2, n_full2)        # 2N 재계산
                else:
                    k2h, v2h = self._attn2_kv(a2, n2_hard)      # 2k fresh
                    ck, cv = cache.self_img_kv[bi]
                    k2 = _scatter_rows(ck.clone(), hard, k2h)
                    v2 = _scatter_rows(cv.clone(), hard, v2h)
                o2 = a2.to_out[0](_attn(q2, k2, v2, self.heads))
                img_hard = img_hard + g_msa2.unsqueeze(1) * o2
            n2h = block.norm2(img_hard) * (1 + sc_mlp[:, None]) + s_mlp[:, None]
            img_hard = img_hard + g_mlp.unsqueeze(1) * block.ff(n2h)
            if o_c is not None and not pre_only:
                ctx = self._ctx_update(block, ctx, o_c, c_gates)

        img_final = _scatter_rows(cache.img_states[-1].clone(), hard, img_hard)
        out = self.t.norm_out(img_final, temb)
        return self.t.proj_out(out)