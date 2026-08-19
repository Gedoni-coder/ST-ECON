# Edge-inference experiments

Implementations written to understand the mechanics behind efficient transformer
inference on constrained hardware. **These are learning exercises, not
contributions.** Every claim below is stated at the level the code actually
supports; where an earlier version of this repository overstated something, the
correction is recorded rather than quietly removed.

## `block_tiled_attention.py`

Tiled multi-head attention with online-softmax rescaling, in the style of
FlashAttention (Dao et al., 2022). Peak activation memory is bounded by the
block size instead of growing with `seq_len²`.

**This is not MAS-Attention.** MAS-Attention (Shakerdargah, Lu, Gao & Niu, MLSys
2025) schedules attention operators across heterogeneous compute units — CPU,
NPU, GPU — overlapping compute and memory movement between them. This file does
single-device block tiling, which is a prerequisite for that work and not the
same contribution. This file was previously named `mas_attention.py`, which was
wrong.

### A bug worth recording

The first version took the softmax maximum *per key block* while sharing
accumulators across all key blocks, so terms computed under different maxima
were summed together. Measured against reference attention:

| seq_len | block | unrescaled error | corrected error |
|---|---|---|---|
| 64 | 32 | 3.81e-01 | 7.77e-16 |
| 128 | 64 | 3.53e-01 | 3.89e-16 |
| 257 | 32 | 2.99e-01 | 5.55e-16 |
| 512 | 128 | 2.62e-01 | 3.33e-16 |
| 64 | 64 | 1.99e-10 | 4.44e-16 |

The last row is the important one. At `seq_len=64` with a block size resolving
to 64, there is a single block and the bug cannot occur — and that was exactly
the configuration the original demo used. It printed an output shape and a
latency and exited cleanly while computing the wrong answer.

The fix is to carry a running maximum and rescale accumulated numerator and
denominator by `exp(m_old − m_new)` before adding each new block.

`tests/test_block_tiled_attention.py` reconstructs the original formulation and
asserts it disagrees with reference attention, so the regression cannot return
silently. It also covers ragged sequence lengths, block-size invariance,
overflow under large activations, and gradient flow.

```bash
pip install -r requirements.txt
python block_tiled_attention.py    # self-check against reference
pytest tests/ -v
```

## `quantize_model.py`, `attention_tiling.cu`

Quantization and a CUDA tiling kernel. Not audited to the same standard as the
above; treat as scratch work.

---

# Corrections to `core/transfusion_accelerator.py`

The following labels in that module described algorithms it does not implement.
The code is retained because the underlying mechanics are useful; the names have
been corrected.

**`OnePassAttention` — correct, and verified.** Running max with rescaling,
agreeing with reference attention to 5e-16 at every tile size tested. It
self-checks on each call by comparing against unfused attention.

**`TileSeek` — random search, not MCTS.** The docstring claimed Monte Carlo Tree
Search. `search()` draws uniformly from a candidate list for a fixed number of
iterations: no tree, no visit counts, no selection policy, no backpropagation.
It also does not converge — five consecutive runs at identical settings returned
`tile_p` of 1, 32, 128, 128 and 64. It is a random sampler over tile shapes
subject to a buffer-size constraint, which is a legitimate baseline but should
be called that.

**`DPipeScheduler` — greedy list scheduling, not dynamic programming.** The
docstring claimed "DAG bipartition + DP overlap." The implementation makes one
forward pass over dictionary insertion order, assigning each node to whichever
of two resource timelines matches its PE type. There is no DP table and no
recurrence.

**It also reports a null result that is worth keeping.** At the default
configuration the scheduler saves 0.002 µs out of 31.3 µs — a speedup of 1.00.
The reason is visible in its own output: the 2D matmul units account for 98.1%
of total latency and the 1D vector units 1.9%. There is almost nothing to
overlap. Two-resource pipelining only pays when the two subgraphs carry
comparable load, and in this transformer configuration they do not. This is a
real finding about when the technique helps, and it should be reported rather
than presented as a speedup.

**Eviction is furthest-last-use, not Belady's MIN.** Belady evicts the line whose
*next* use is furthest away. The implementation ranks by the index of each
tensor's *last* reader, which differs whenever a tensor has multiple readers. It
is a reasonable approximation; it is not the optimal policy and should not be
named for it. The tracker does behave sensibly at the boundaries: 61.1% on-chip
retention at 64 KB, and 0% at a 1-byte budget.

**Paper attribution.** The module docstring referred to "the MAS-Attention Paper
2." TransFusion (MICRO 2025) and MAS-Attention (MLSys 2025) are separate papers
with separate contributions. The mechanisms modelled here — Einsum cascade DAG,
two-resource scheduling, tile search, one-pass attention — follow TransFusion.
