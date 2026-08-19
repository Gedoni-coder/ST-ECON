"""
Correctness tests for BlockTiledAttention.

The point of this file is `test_original_formulation_was_wrong`, which
reconstructs the buggy per-block-max version and asserts that it disagrees with
reference attention. If a future refactor reintroduces that mistake, the other
tests here fail. If someone deletes the rescaling step believing it redundant,
this file explains why it is not.

Run:  pytest tests/ -v
"""
import math

import pytest
import torch
import torch.nn.functional as F

from block_tiled_attention import BlockTiledAttention

TOL = 1e-4


def _make(d_model=128, n_heads=4, sram=16384, seed=0):
    torch.manual_seed(seed)
    return BlockTiledAttention(d_model=d_model, n_heads=n_heads, sram_limit_bytes=sram)


# --------------------------------------------------------------------------
# The regression test: the original implementation, and proof it was wrong.
# --------------------------------------------------------------------------

def _buggy_forward(model, x):
    """The original implementation, verbatim in behaviour.

    Per-block max, shared accumulators, no rescaling. Kept so the failure it
    caused stays visible and reproducible rather than becoming folklore.
    """
    B, S, _ = x.shape
    H, dk = model.n_heads, model.d_k

    def heads(t):
        return t.view(B, S, H, dk).transpose(1, 2)

    Q, K, V = heads(model.q_proj(x)), heads(model.k_proj(x)), heads(model.v_proj(x))
    block = model.block_size(x.element_size())
    out = torch.zeros_like(Q)

    for q0 in range(0, S, block):
        q1 = min(q0 + block, S)
        Qb = Q[:, :, q0:q1, :]
        num = torch.zeros(B, H, q1 - q0, dk)
        den = torch.zeros(B, H, q1 - q0, 1)
        for k0 in range(0, S, block):
            k1 = min(k0 + block, S)
            scores = torch.matmul(Qb, K[:, :, k0:k1, :].transpose(-2, -1)) / math.sqrt(dk)
            exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
            num = num + torch.matmul(exp_scores, V[:, :, k0:k1, :])
            den = den + exp_scores.sum(dim=-1, keepdim=True)
        out[:, :, q0:q1, :] = num / (den + 1e-9)

    out = out.transpose(1, 2).contiguous().view(B, S, model.d_model)
    return model.out_proj(out)


def test_original_formulation_was_wrong():
    """Multi-block sequences expose the missing rescaling."""
    model = _make()
    x = torch.randn(2, 512, 128)
    assert model.block_size(x.element_size()) < 512, "test needs more than one block"

    err = (_buggy_forward(model, x) - model.reference_forward(x)).abs().max().item()
    assert err > 1e-2, (
        f"expected the unrescaled version to be visibly wrong, got {err:.2e}"
    )


def test_original_bug_is_invisible_at_one_block():
    """Why it went unnoticed: the old test used seq_len=64, a single block."""
    model = _make()
    x = torch.randn(2, 64, 128)
    assert model.block_size(x.element_size()) >= 64, "this case must be single-block"

    err = (_buggy_forward(model, x) - model.reference_forward(x)).abs().max().item()
    assert err < TOL, "single-block case should hide the bug — that was the trap"


# --------------------------------------------------------------------------
# Correctness of the fixed implementation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len", [64, 128, 257, 512])
def test_matches_reference_across_lengths(seq_len):
    model = _make()
    x = torch.randn(2, seq_len, 128)
    err = (model(x) - model.reference_forward(x)).abs().max().item()
    assert err < TOL, f"seq_len={seq_len}: max error {err:.3e}"


@pytest.mark.parametrize("sram", [1024, 4096, 16384, 65536])
def test_block_size_does_not_change_output(sram):
    """Tiling is an execution strategy; results must be invariant to it."""
    model = _make(sram=sram)
    x = torch.randn(2, 384, 128)
    err = (model(x) - model.reference_forward(x)).abs().max().item()
    assert err < TOL, f"sram={sram} (block={model.block_size(x.element_size())}): {err:.3e}"


def test_ragged_sequence_length():
    """Sequence length not divisible by block size — partial final tile."""
    model = _make(sram=1024)
    x = torch.randn(1, 333, 128)
    err = (model(x) - model.reference_forward(x)).abs().max().item()
    assert err < TOL


def test_large_score_magnitudes():
    """Inputs scaled up so exp() would overflow without max subtraction."""
    model = _make()
    x = torch.randn(2, 256, 128) * 50.0
    out = model(x)
    assert torch.isfinite(out).all(), "overflow or NaN under large activations"
    err = (out - model.reference_forward(x)).abs().max().item()
    assert err < 1e-2


def test_gradients_flow():
    model = _make()
    x = torch.randn(2, 256, 128, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rejects_bad_shapes():
    model = _make()
    with pytest.raises(ValueError):
        model(torch.randn(2, 64))          # missing batch or feature dim
    with pytest.raises(ValueError):
        model(torch.randn(2, 64, 64))      # wrong d_model


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        BlockTiledAttention(d_model=128, n_heads=5)
