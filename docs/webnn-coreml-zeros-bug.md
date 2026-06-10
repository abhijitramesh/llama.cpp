# Draft bug report: WebNN (CoreML backend) returns all zeros for mask-less GQA attention subgraph

**Status: draft for issues.chromium.org — NOT yet filed.**

## Summary

A matmul→softmax→matmul attention-shaped MLGraph with grouped-query
broadcasting (rank-5) and **no additive mask** silently returns all-zero
outputs for specific shapes on Chrome's WebNN CoreML backend
(MLComputeUnitsAll), while the same graph with a mask input, or with other
head sizes, computes correctly. No error is surfaced — the dispatch succeeds.

## Environment

- Chrome 149.0.7827.54, macOS (Darwin 25.5.0), Apple M3
- `--enable-features=WebMachineLearningNeuralNetwork` (CoreML backend with
  MLComputeUnitsAll — verified via `--webnn-coreml-dump-model` logging)

## Failing shape (from llama.cpp test-backend-ops FLASH_ATTN_EXT matrix)

- q: float32 [1, 4, nb, 128] reshaped to [1, 1, 4, nb, 128] (GQA group 4)
- k,v: float16 [1, 1, kv=512, 128] reshaped to [1, 1, 1, 512, 128], cast to f32
- graph: `matmul(q5, transpose(k5, [0,1,2,4,3]))` → `linear(alpha=1/sqrt(128))`
  → `softmax(axis 4)` → `matmul(probs, v5)` → `transpose` → output
- nb = 3, mask ABSENT. Result: output tensor is entirely zeros
  (NMSE vs CPU reference ≈ 1.0).
- The identical graph **with** an additive f16 mask input computes correctly,
  as do hsk=64/256 variants and all masked GQA variants
  (855+ passing cases in the same harness).

## Repro sketch

Build the five-op graph above with MLGraphBuilder, dispatch with random
f32/f16 inputs in [-1, 1], read the output: all zeros. A standalone HTML
repro can be distilled from
`llama.cpp` branch `webnn-prototype`, `tests/test-webnn.cpp`
(`flash_attn_gqa` case generalizes; the failing variant is hsk=128, mask=0)
and `run-backend-ops.html` (`test-backend-ops test -b WebNN -o FLASH_ATTN_EXT`).

## Suspected mechanism

Softmax over a large unmasked logit row in reduced precision producing
inf/NaN that collapses to zeros, or a CoreML compute-unit fallback with
divergent precision (cf. webmachinelearning/webnn#778 silent-fallback class).

## Workaround in our backend

Mask-less FLASH-attention shapes are gated off in `supports_op`
(ggml/src/ggml-webnn/ggml-webnn.cpp); llama-class models always supply masks.
