---
id: kernel-small-m-m-grouped-gemm
title: Small-M M-grouped GEMM on SM90
type: kernel
architectures: [sm90]
tags: [grouped-gemm, moe, gemm, tma, wgmma, persistent-kernel, tile-scheduling]
confidence: inferred
reproducibility: snippet
kernel_types: [grouped-gemm, gemm, moe]
languages: [cuda-cpp]
aliases: [tiny-M M-grouped GEMM, ragged M-grouped GEMM, small-M grouped GEMM]
related: [kernel-grouped-gemm, kernel-deepgemm, technique-tile-scheduling, technique-persistent-kernels, pattern-tail-effect, pattern-memory-bound]
sources: [blog-deepgemm, pr-DeepGEMM-88, pr-DeepGEMM-168, pr-DeepGEMM-304, pr-cutlass-2790, pr-sglang-11432]
performance_claims: []
blackwell_relevance: CUTLASS has an SM100 ragged-contiguous grouped GEMM, but its layouts and descriptor ownership must be re-derived instead of copying SM90 WGMMA tactics.
---

# Small-M M-grouped GEMM on SM90

Small-M means that each group's useful rows are thin relative to its legal CTA
tile; it is not one universal threshold. Padding, weight traffic, and tail waves
can then dominate useful MMA work.

## Evidence boundary

DeepGEMM documents contiguous and masked M-grouped layouts. SGLang PR 11432
reports unnecessary computation when a large tiled-MMA shape serves experts
with very few tokens. DeepGEMM PRs 88 and 168 treat B multicast and scheduling
as performance choices for M-grouped contiguous GEMM.

These sources establish mechanisms, not a universal tile, stage count, dispatch
threshold, or speedup. The sequence below is inferred and must be measured for
the target dtype, routing distribution, and GPU.

## Retained SM90 anchor

DeepGEMM PR 304 retains the SM90 BF16 dispatcher, grouped scheduler, and tuning
heuristic. This contiguous excerpt from its pinned `heuristics/sm90.hpp` shows
that expected group shape is converted into blocks, waves, and last-wave use:

```cpp
const auto num_blocks =
    ceil_div(desc.get_expected_m(), layout.block_m) *
    ceil_div(desc.get_expected_n(), layout.block_n) *
    desc.get_expected_num_groups();
const auto num_waves = ceil_div(num_blocks, desc.num_sms);
const auto num_last_blocks = num_blocks % desc.num_sms;
const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;
```

The same retained heuristic jointly enumerates block M/N, cluster shape, and
pipeline legality; its cost model includes L1/L2 traffic and rejects multicast
when the workload has only one wave. The excerpt is not a standalone kernel.

## Optimization order

| Evidence | Candidate to benchmark |
|---|---|
| Layout conversion or padding dominates | Reuse a compatible contiguous layout; use masked layout when stable graph shapes matter |
| Valid M is much smaller than the legal M tile | Select the smallest legal M tile, then retune N tile, stages, and cluster together |
| Last-wave utilization is low | Change the complete tile/cluster tactic or SM budget; do not tune grid size alone |
| B/weight traffic dominates | Keep same-group work close and test B multicast only when cluster and wave legality hold |
| Group traversal is expensive | Compare persistent mappings while preserving exact coverage of every `(group, m_tile, n_tile)` |

For contiguous DeepGEMM layouts, block M is constrained by the queried layout
alignment; masked layouts have a different candidate set. Do not transplant a
dense-GEMM small-M tile without checking that contract. Keep routing-dependent
selection on device unless the caller supplies a stable hint.

## Acceptance gate

1. Cover empty groups, nonuniform M, M/N tails, and layout-alignment boundaries;
   use output canaries and verify input immutability.
2. Time the physical operator boundary, including required metadata and layout
   conversion; compare implementations at the same boundary.
3. Use paired GPU-event timing over multiple routing replicas, then inspect
   traffic and wave counters only for candidates that first improve latency.
4. Keep only mechanism-backed dispatch rules and fall back to the general
   grouped kernel outside validated coverage.

## Blackwell boundary

CUTLASS PR 2790 is the retained SM100 ragged-contiguous reference and avoids
per-group tensor-map updates for the shared-shape weight operand. Transfer the
diagnostic order, then re-derive SM100 layouts, descriptors, and schedules.
