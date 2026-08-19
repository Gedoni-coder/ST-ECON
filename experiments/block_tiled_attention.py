"""
Block-tiled attention with online-softmax rescaling.

WHAT THIS IS
------------
A from-scratch PyTorch implementation of tiled attention in the style of
FlashAttention (Dao et al., 2022): queries and keys are processed in blocks so
that peak activation memory stays bounded by the block size rather than growing
with seq_len^2. It runs on a single device.

WHAT THIS IS NOT
----------------
This is NOT MAS-Attention. MAS-Attention (Shakerdargah, Lu, Gao & Niu, MLSys
2025) schedules attention operators across *heterogeneous* compute units — CPU,
NPU, GPU — overlapping compute and memory movement across them. That is a
scheduling contribution across devices. This file only does single-device block
tiling, which is a prerequisite for that work but is not the same thing. An
earlier version of this file carried the MAS-Attention name; that was wrong and
has been corrected.

HISTORY: THE BUG THIS FILE USED TO HAVE
----------------------------------------
The first version computed, inside the key loop:

    exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True))
    accum_num += exp_scores @ V_block
    accum_den += exp_scores.sum(dim=-1, keepdim=True)

The max was taken *per key block*, while the accumulators were shared across all
key blocks. Every block therefore subtracted a different constant before
exponentiating, and the sums combined terms on incompatible scales. Against
reference attention the output was wrong by up to 0.73 in absolute value.

It went unnoticed because the module's only test used seq_len=64 with a block
size that resolved to 64 — a single block, in which the bug cannot appear. The
test printed a shape and a latency and exited cleanly.

The fix is online-softmax rescaling: carry a running max `m`, and whenever it
increases, rescale the accumulators already collected by exp(m_old - m_new)
before adding the new block's contribution. See `_rescale` below.

`tests/test_block_tiled_attention.py` reproduces the failure on the old
formulation and asserts agreement with reference attention across several block
sizes and sequence lengths.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockTiledAttention(nn.Module):
    """Multi-head attention computed in tiles under a fixed activation budget.

    Args:
        d_model: model dimension.
        n_heads: number of attention heads; d_model must be divisible by it.
        sram_limit_bytes: activation budget used to derive the block size. This
            is a *modelling* parameter — it determines tiling, it does not
            measure or enforce real on-chip memory.
    """

    def __init__(self, d_model: int, n_heads: int, sram_limit_bytes: int = 65536):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sram_limit_bytes = sram_limit_bytes

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def block_size(self, element_size: int) -> int:
        """Largest block whose Q and K tiles fit the activation budget."""
        return max(8, self.sram_limit_bytes // (self.d_k * element_size * 2))

    @staticmethod
    def _rescale(num, den, running_max, new_max):
        """Bring accumulators collected under `running_max` onto `new_max`.

        This is the step whose absence made the original implementation wrong.
        exp(m_old - m_new) <= 1, so this shrinks earlier contributions to match
        the new, larger maximum before the current block is added.
        """
        scale = torch.exp(running_max - new_max)
        return num * scale, den * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected [B, S, {self.d_model}], got {tuple(x.shape)}")

        B, S, _ = x.shape
        H, dk = self.n_heads, self.d_k

        def heads(t):
            return t.view(B, S, H, dk).transpose(1, 2)  # [B, H, S, dk]

        Q, K, V = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))

        block = self.block_size(x.element_size())
        out = torch.empty_like(Q)

        for q0 in range(0, S, block):
            q1 = min(q0 + block, S)
            Qb = Q[:, :, q0:q1, :]

            # Running statistics for this query block, per query row.
            running_max = torch.full((B, H, q1 - q0, 1), float("-inf"),
                                     device=x.device, dtype=Q.dtype)
            num = torch.zeros(B, H, q1 - q0, dk, device=x.device, dtype=Q.dtype)
            den = torch.zeros(B, H, q1 - q0, 1, device=x.device, dtype=Q.dtype)

            for k0 in range(0, S, block):
                k1 = min(k0 + block, S)
                scores = torch.matmul(Qb, K[:, :, k0:k1, :].transpose(-2, -1)) / math.sqrt(dk)

                block_max = scores.max(dim=-1, keepdim=True).values
                new_max = torch.maximum(running_max, block_max)

                # Rescale what we already have, THEN add this block.
                num, den = self._rescale(num, den, running_max, new_max)
                exp_scores = torch.exp(scores - new_max)
                num = num + torch.matmul(exp_scores, V[:, :, k0:k1, :])
                den = den + exp_scores.sum(dim=-1, keepdim=True)
                running_max = new_max

            out[:, :, q0:q1, :] = num / den

        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)

    @torch.no_grad()
    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unfused attention over the same projections, for correctness checks."""
        B, S, _ = x.shape
        H, dk = self.n_heads, self.d_k

        def heads(t):
            return t.view(B, S, H, dk).transpose(1, 2)

        Q, K, V = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BlockTiledAttention(d_model=128, n_heads=4, sram_limit_bytes=4096)
    x = torch.randn(2, 512, 128)  # 512, not 64 — long enough to force many blocks

    print(f"seq_len=512, block_size={model.block_size(x.element_size())}")
    tiled = model(x)
    ref = model.reference_forward(x)
    err = (tiled - ref).abs().max().item()
    print(f"max abs error vs reference: {err:.3e}")
    print("PASS" if err < 1e-4 else "FAIL")
