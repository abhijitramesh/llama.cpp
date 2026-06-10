# Browser LLM inference on Apple Silicon: WebGPU vs WebNN vs Hybrid

**Final comparison — Apple M3 MacBook, Chrome 149 (headless), June 2026.**
Branch `webnn-prototype`. llama-bench, `-p 128 -n 64 -r 2`, warmup runs amortize
graph compiles. WebNN uses `-fa 1`; WebGPU/CPU use FA auto (the wasm WebGPU
backend lacks subgroup-matrix FA, so forcing it would push attention to CPU).
Power via `sudo powermetrics` (system-wide; idle floor ~4 W — deltas matter).
Energy/token = (CPU+GPU+ANE avg power) / throughput. Raw data:
`bench-results-final/`.

WebNN backend = this branch's ggml-webnn (pipelined dispatch, device-resident
KV via scatterND, weights as dequantizeLinear constants, chunked CoreML
compilation). "WebNN best" = chunk=24 + prune (CoreML delegate; verified the
default backend on this machine is CoreML with MLComputeUnitsAll).

## SmolLM2-135M-Instruct, Q4_0 (87 MB) — primary

| backend                  | pp128 t/s | tg64 t/s | CPU mW | GPU mW | ANE mW | prefill mJ/t | decode mJ/t |
|--------------------------|-----------|----------|--------|--------|--------|--------------|-------------|
| wasm CPU (1 thread)      | 23        | 21       | 8129   | 0      | 0      | 357          | 395         |
| pure WebNN (best)        | **1950**  | 47       | 9348   | 0      | 0      | **4.8**      | 198         |
| pure WebGPU              | 1523      | **174**  | 6054   | 718    | 0      | 4.4          | **39**      |
| hybrid (composed: WebNN prefill + WebGPU decode) | **1950** | **174** | mixed | mixed | 0 | **4.8** | **39** |

## Qwen2.5-0.5B-Instruct, Q4_0 (429 MB) — scale point

| backend             | pp128 t/s | tg64 t/s | CPU mW | GPU mW | ANE mW | prefill mJ/t | decode mJ/t |
|---------------------|-----------|----------|--------|--------|--------|--------------|-------------|
| wasm CPU            | 7.4       | 6.4      | 7464   | 0      | 0      | 1008         | 1166        |
| pure WebNN (best)   | 734–886   | 17       | 9161   | 0      | 0      | ~11          | 539         |
| pure WebGPU         | 771       | **98**   | 5506   | 2001   | 0      | 9.7          | **77**      |
| hybrid (composed)   | ~886      | **98**   | mixed  | mixed  | 0      | ~11          | **77**      |

f16 references (SmolLM2): WebGPU 1670/129 (5.8 W total); WebNN-NPU 722/10.0
with the ANE engaged at 251 mW — the only configuration that touches the ANE,
and the slowest.

## Verdicts

1. **WebNN wins prefill on quantized models — at short-to-medium prompt
   lengths.** 1950 vs 1523 t/s on SmolLM2-Q4_0 at pp128 (+28%) and parity at
   Qwen-0.5B — the first outright WebNN throughput wins over WebGPU in this
   project. CoreML's compiled graphs with compressed int4 constants beat
   hand-rolled WGSL at batch compute. Energy per prefill token is at parity
   (~4.4–4.8 mJ). Scope caveat (measured later at pp1024): WebGPU scales
   better with prompt length — ~2300 vs WebNN's ~1440 t/s at 1024 tokens, as
   its fixed per-dispatch overhead amortizes while the CoreML int4 path
   saturates; the crossover lies between 128 and 1024 tokens. Cold-start
   TTFT is always WebGPU's: WebNN re-pays CoreML graph compilation every
   page load until graph caching ships.
2. **WebGPU owns decode**: 3.7–5.8x WebNN throughput and ~5–7x better decode
   energy. WebNN decode (47 t/s) still beats browser CPU by >2x.
3. **The hybrid is real and simple**: route prefill to WebNN, decode to
   WebGPU. Composed from measured phases (the one-time KV handover is ~1.7 MB
   ≈ low single-digit ms, negligible against a 66 ms prefill). It delivers the
   best column of each: +28% faster time-to-first-token than pure WebGPU at
   equal decode speed. Today it requires two model instances or weight
   duplication: a combined binary runs both backends but weights live on one
   backend's buffers (WebGPU needs device weights, WebNN host weights), and
   naive scheduler mixing fragments the graph (62 splits, slower than either).
   The shipping path is the MLTensor<->WebGPU zero-copy interop
   (createContext(gpuDevice)) prototyped in Chromium but not yet released.
4. **The ANE is an honest negative for LLM decode**: f16-only by architecture,
   engaged at 251 mW only in the slowest configuration; CoreML routes the fast
   quantized path to CPU/GPU compute units instead. Consistent with published
   ANE research (dispatch-bound, conv-optimized, unreachable fast paths).
   Its published niche - batched prefill energy - is made moot here by WebNN
   prefill already being energy-competitive on this stack.
5. **The GGUF-native differentiator held up**: Q4_0, Q8_0 and Q4_K (first
   K-quant on WebNN anywhere; ONNX cannot express it - onnx#7691) all run
   bit-faithfully as dequantizeLinear constants, with visibly better Q4_K_M
   output quality. 256-divisible rows required for Q4_K; others fall back.
6. **Browser reality check**: everything here needs
   `--enable-features=WebMachineLearningNeuralNetwork` (+ CoreML compute-unit
   flags for explicit device selection); no graph caching exists, so every
   page load recompiles (CoreML: seconds per 24-node chunk, minutes for
   monolithic graphs); no dynamic shapes, so graphs are static per sequence
   bucket. Chromium-only.

## Reproduction

```
# build (emsdk 5.x on PATH)
emcmake cmake -B build-webnn  -DGGML_WEBNN=ON  -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
emcmake cmake -B build-webgpu -DGGML_WEBGPU=ON -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-webnn  --target llama-bench test-webnn test-backend-ops -j
cmake --build build-webgpu --target llama-bench -j
# serve build-*/bin over HTTP, open bench.html with query knobs:
#   model=, args= (llama-bench argv), chunk=24, prune=1, nosc=1, webnn=npu,
#   f16=1, minb=/maxb= (hybrid routing window)
# Chrome: --enable-features=WebMachineLearningNeuralNetwork[,WebNNCoreML,
#   WebNNCoreMLExplicitGPUOrNPU]
```

Validation status at time of writing: tests/test-webnn 54/54,
test-backend-ops -b WebNN 1638/1638, end-to-end generation verified on
stories15M/110M, SmolLM2-135M (f16/Q4_0/Q4_K_M) and Qwen2.5-0.5B (f16/Q4_0).
