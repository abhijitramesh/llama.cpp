# WebGPU vs WebNN benchmark — Apple M3 MacBook, Chrome 149 (headless)

## Pure-WebNN steady state with quantized weights (commit 364727ce7)

SmolLM2-135M, llama-bench -p 128 -n 64 -r 2 -fa 1 (warmup amortizes
graph compiles). Q4_0 weights are dequantizeLinear graph constants.

| config                          | pp128 t/s | tg64 t/s | ANE mW avg |
|---------------------------------|-----------|----------|------------|
| LiteRT split-mode, f16          | 699       | 14.7     | 0          |
| LiteRT split-mode, **q4_0**     | 884       | **27.8** | 0          |
| CoreML chunk=24, f16            | 664       | 9.5      | 228        |
| **CoreML chunk=24, q4_0**       | **1528**  | **38.1** | 0          |
| CoreML chunk=24, q4_0 + f16 cast| 479       | 5.8      | 78         |
| (reference: WebGPU, f16 model)  | 1640      | 118      | 0          |
| (reference: browser CPU, q4_0)  | ~18 wall  |          | 0          |

Findings:

1. **Quantization roughly doubles decode on every delegate**
   (LiteRT 14.7 -> 27.8, CoreML 9.5 -> 38.1): compressed-constant
   execution delivers the bandwidth win, and WebNN decode now beats
   browser CPU (~2x) for the first time.
2. **CoreML + chunked graphs + Q4_0 is the fastest pure-WebNN config**:
   prefill reaches 93% of WebGPU-f16 and decode 32%. The chunking that
   unlocked CoreML compilation pays off fully once weights are
   constants.
3. **The ANE remains f16-bound and slow for LLM shapes**: it engages
   for f16 graphs (228 mW) at the lowest throughput, and CoreML's fast
   quantized path bypasses it entirely (0 mW). CORRECTION after
   literature review: that fast q4 path runs on the **Metal GPU**, not
   CPU - Apple's docs route per-block int4 dequantizeLinear to GPU,
   and 38 t/s is GPU-class. Forcing f16 compute over q4 constants
   re-engages the ANE weakly (78 mW) and tanks throughput. Validated
   against published ANE research (Orion paper, Apple docs): ANE is
   architecturally f16-only, ~0.1 ms dispatch overhead per op,
   matmul 3x slower than conv on ANE, and its fast paths
   (conv2d-reshaped attention) are unreachable via WebNN. Consensus:
   ANE suits encoders/batched prefill, GPU suits decode.
   Delegate-naming caveat: Chromium source has kWebNNCoreML
   default-ON for Apple Silicon; our plain-flag runs behaved like the
   in-process TFLite/XNNPACK CPU fallback (fast compiles, no
   GPU/ANE power). "LiteRT" rows = whatever the plain WebNN flag
   provides on this box; delegate identity deserves a dump-model
   verification.
4. Implementation lesson recorded in the commit: baked constants carry
   tensor identity into compiled graphs, so the graph-cache key must
   include the constant pointers - same-shaped weights across layers
   otherwise silently share one layer's weights.

TODO for the three-way story: WebGPU q4_0 reference numbers, larger
GQA models (Qwen 0.5B q4_0), energy/token table across all configs.

## External validation (June 2026, four-agent literature/issue review)

Validated against Chromium source, W3C WG minutes, ORT WebNN EP, and
published ANE research:

- Our architecture independently converged on the ecosystem standard:
  ORT's WebNN EP also bakes int4 weights as dequantizeLinear constants
  (MatMulNBits decomposition), writes KV via scatterND, and ping-pongs
  preallocated KV MLTensors. No prior ggml->WebNN art exists; the
  GGUF-native path is novel and uniquely K-quant-capable (ONNX cannot
  represent GGUF K-quants, onnx#7691).
- Per-dispatch IPC overhead (0.5-2 ms) and CoreML's minutes-long
  compile of large graphs (no caching in Chromium - temp dir per
  build, webnn#807 open) are documented architecture, validating the
  chunked-compilation approach (~24-node chunks near the sweet spot).
- Documented patterns we should adopt: dispatch() and writeTensor()
  are fire-and-forget - queue all dispatches and await ONE final
  readTensor; chain chunk outputs as next-chunk MLTensor inputs
  instead of host roundtrips. We currently await per dispatch.
- Element-wise scatterND is CPU-emulated on the TFLite path (no native
  op) - explains the transposed-V scatter pathology.
- The mask-less GQA flash-attention zeros on the default delegate has
  no filed bug upstream; worth reporting with a minimal repro.
- Hybrid correction from published work: the winning phase split is
  ANE/NPU for PREFILL (batched, 282x GPU power reduction) + GPU for
  DECODE - the reverse of our earlier sketch. Zero-copy
  WebNN<->WebGPU interop (createContext(gpuDevice)) is prototyped in
  Chromium for CoreML but unshipped, so phase-level splitting (one
  boundary) remains the right hybrid granularity.
- Dynamic shapes (webnn#883) and graph caching (webnn#807) are
  2026-roadmap items, unshipped: pre-bucketed static graphs and
  per-load recompiles are unavoidable today.

## Cross-model sweep (v3 + GQA, commit 97e9bb3af)

Four f16 models spanning architecture families (llama + qwen2), MHA and
GQA, 6-30 layers. Configs: wasm CPU, WebNN default (LiteRT), WebNN
npu + force-f16 (CoreML/ANE), WebGPU. All models generate correct text
end-to-end with attention/ROPE on WebNN.

Prefill pp128 (t/s):

| model                       | cpu  | webnn | webnn-npu | webgpu | webnn/webgpu | npu ANE mW |
|-----------------------------|------|-------|-----------|--------|--------------|------------|
| stories15M (MHA, 6L)        | 320  | 4343  | 4511      | 8918   | 51%          | 353        |
| stories110M (MHA, 12L)      | 24   | 819   | **1026**  | 2855   | 36%          | 504        |
| SmolLM2-135M (GQA 9/3, 30L) | 18   | 590   | **651**   | 1640   | 40%          | 260        |
| Qwen2.5-0.5B (GQA 14/2, 24L)| 5.7  | 162   | **276**   | 876    | 32%          | 291        |

Decode tg64 (t/s):

| model         | cpu  | webnn | webnn-npu | webgpu |
|---------------|------|-------|-----------|--------|
| stories15M    | 124  | 77    | 67        | 329    |
| stories110M   | 18   | 18    | 18        | 176    |
| SmolLM2-135M  | 14   | 15    | 4.2       | 118    |
| Qwen2.5-0.5B  | 4.1  | 3.4   | 3.0       | 55     |

Cross-model findings:

1. **WebNN prefill is universally strong**: 14-49x wasm CPU and 32-51%
   of WebGPU across every model and architecture tested.
2. **The ANE advantage grows with model size**: npu-f16 beats LiteRT by
   1.04x at 15M, 1.25x at 110M, 1.7x at Qwen-0.5B — bigger matmuls
   amortize CoreML dispatch and favor the NPU, exactly the trend the
   heterogeneous (ANE+GPU) thesis needs. ANE power 260-504 mW sustained.
3. **Decode is the open front everywhere**: WebNN decode tracks wasm CPU
   (within noise) because per-layer KV-write splits + materialize-all
   host writeback dominate; CoreML's higher dispatch latency makes
   npu decode worst on the 30-layer SmolLM2 (4.2 t/s). This is the v4
   workload (scatter KV writes in-graph, device-resident MLTensors).
4. WebGPU GPU power scales with model (128 -> 1921 mW at 0.5B) and it
   stays the overall throughput/energy champion (~9 vs ~36 mJ/token
   prefill at 0.5B for WebNN-NPU; wasm CPU is ~1600).

## v3 backend: attention on WebNN (commit cdc355ab6)

ROPE, FLASH_ATTN_EXT and strided (permuted-dense) views now translate
into the compiled MLGraphs, so whole transformer layers run on WebNN.
Graph splits dropped 27 -> 13 at decode (only the KV-cache SET_ROWS
writes remain on CPU). llama's flash-attn auto-detection resolves to
the non-FA path on this backend; `-fa 1` (composed flash-attn) is
slightly faster still (15M: pp 5619).

stories15M (f32 model except f16 rows):

| config        | pp128 t/s | tg64 t/s | CPU mW | GPU mW | ANE mW        |
|---------------|-----------|----------|--------|--------|---------------|
| cpu           | 1061      | 540      | 7250   | 100    | 0             |
| webnn-default | 4747      | 110      | 7725   | 122    | 0             |
| webnn-npu-f16 | 4595      | 66       | 8918   | 65     | **351 (pk 875)** |
| webgpu        | 8767      | 264      | 7450   | 179    | 0             |
| webgpu-f16    | 8713      | 329      | 7689   | 214    | 0             |

stories110M:

| config        | pp128 t/s | tg64 t/s | CPU mW | GPU mW | ANE mW        |
|---------------|-----------|----------|--------|--------|---------------|
| cpu (f32)     | 120       | 77       | 9115   | 105    | 0             |
| cpu-f16       | 24        | 18       | 9304   | 105    | 0             |
| webnn-default | 934       | 18       | 9794   | 81     | 0             |
| webnn-npu-f16 | **1007**  | 18       | 9032   | 115    | **507 (pk 788)** |
| webgpu        | 2866      | 112      | 6363   | 1031   | 0             |
| webgpu-f16    | 2826      | 185      | 6538   | 828    | 0             |

v3 findings:

1. **Prefill gap to WebGPU halved again**: 15M 4747 vs 8767 (54%, was
   26% in v2, 13% in v1); 110M 934-1007 vs ~2850 (~35%, was 14%).
   Versus wasm CPU the 110M prefill advantage is 8x (f32) / 42x (f16).
2. **The ANE now runs the whole attention stack**: webnn-npu-f16 at
   110M sustains 507 mW ANE across the run (peak 788) and is the
   fastest WebNN configuration at that size - the first time the
   NPU path wins outright over LiteRT-CPU.
3. **Decode is unchanged (~110 / ~18 t/s)** and is now cleanly
   attributable: 13 graph splits per token (KV SET_ROWS on CPU between
   every layer) plus the materialize-everything host writeback. The
   next levers are exactly the planned v4 items: SET_ROWS/scatter in
   graph, device-resident MLTensors, and output pruning.
4. WebGPU remains ahead on every throughput and energy metric
   (110M prefill ~2.3 vs ~9.4 mJ/token), but the heterogeneous endgame
   (ANE matmuls + GPU attention) is now architecturally within reach:
   both backends compile whole subgraphs and the scheduler can split
   between them.

## v2 backend: whole-graph compilation + f16 (commit cf6725047 + 36de1d7dc)

Same protocol as v1 below. f16 rows use `stories15M-f16.gguf` (converted
with llama-quantize); `*-f16` WebNN rows additionally force f16 matmul
compute (`GGML_WEBNN_FORCE_F16`) and enable Chrome's CoreML delegate.

| config        | pp128 t/s | tg64 t/s | GPU util avg/max % | CPU mW | GPU mW | ANE mW    |
|---------------|-----------|----------|--------------------|--------|--------|-----------|
| cpu (f32)     | 1059      | 541      | 7/8                | 6457   | 95     | 0         |
| cpu-f16       | 324       | 128      | 6/8                | 7603   | 97     | 0         |
| webnn-default | 2319      | 112      | 7/8                | 7177   | 90     | 0         |
| webnn-cpu     | 2351      | 112      | 7/10               | 7008   | 94     | 0         |
| webnn-gpu     | 1932      | 64       | 9/29               | 6711   | 111    | 0         |
| webnn-npu     | 2357      | 118      | 0/0                | 7156   | 0      | 0         |
| webnn-gpu-f16 | 2026      | 69       | 7/24               | 6907   | 124    | 0         |
| webnn-npu-f16 | 2218      | 67       | 8/14               | 7401   | 102    | **376 (peak 926)** |
| webgpu        | 9002      | 331      | 28/71              | 5502   | 205    | 0         |
| webgpu-f16    | 8935      | 323      | 32/67              | 5264   | 215    | 0         |

v2 findings:

1. **Whole-graph compilation works**: vs the v1 per-op prototype, WebNN
   prefill doubled (1213 -> 2319 t/s) and decode improved 3.4x
   (33 -> 112 t/s). Real llama splits fuse ~6.5 ops per MLGraph dispatch.
2. **The ANE engaged** — the headline result. `webnn-npu-f16` (f16
   weights + f16 matmul compute + CoreML delegate + deviceType=npu) is
   the only configuration in the entire investigation with nonzero ANE
   power: 376 mW average, 926 mW peak. WebNN demonstrably reaches
   silicon WebGPU cannot. Every other config (including npu with f32) is
   0 mW: CoreML only routes f16-friendly graphs to the ANE.
3. **ANE != faster here**: tg 67 t/s via the ANE vs 112 on WebNN's own
   CPU path and 331 on WebGPU. At 15M params with host roundtrips per
   split, ANE dispatch latency dominates; the result is an existence
   proof and a power-efficiency avenue, not a throughput win at this
   scale.
4. WebGPU is unchanged on top (9002/331) and still the lowest system
   power; f32 vs f16 model makes little difference for it on this
   bandwidth-light model.
5. `cpu-f16` collapses to 324/128 t/s (generic wasm CPU has no fast f16
   path), which makes WebNN-LiteRT (2351/112) genuinely competitive for
   f16 models in pure-wasm environments — a real niche where WebNN
   already pays off today.

Remaining caveats: tiny model; decode rankings shift at larger scale;
KV-length bucketing for true static-shape reuse and device-resident
tensors (skip host writeback per split) are the next levers.

Model: `tinyllamas/stories15M.gguf` (f32, 24.41M params, 93 MiB).
`llama-bench -p 128 -n 64 -r 3` via wasm/JSPI builds, 2026-06-09.
Branch: `webnn-prototype` (WebNN backend = per-op dispatch prototype,
host buffers; WebGPU backend = upstream, device-resident, -ngl 99).

## Throughput + utilization

| config        | pp128 t/s | tg64 t/s | GPU util avg/max % | Chrome CPU avg/max % | notes |
|---------------|-----------|----------|--------------------|----------------------|-------|
| cpu (wasm, 1 thread) | 1057 | 539  | 8/10  | 82/173  | WebNN disabled |
| webnn-default | 1139      | 34       | 8/20               | 107/142              | LiteRT (TFLite) delegate |
| webnn-cpu     | 1219      | 35       | 7/10               | 105/144              | deviceType=cpu |
| webnn-gpu     | 1227      | 33       | 8/11               | 97/129               | deviceType=gpu, LiteRT |
| webnn-npu     | 1167      | 32       | 7/13               | 106/139              | deviceType=npu, LiteRT |
| webgpu        | **8952**  | **272**  | 17/75              | 84/132               | Metal via Dawn/ANGLE |

With Chrome's CoreML delegate explicitly enabled
(`--enable-features=...,WebNNCoreML,WebNNCoreMLExplicitGPUOrNPU`):

| config    | pp128 t/s | tg64 t/s | GPU util avg/max % |
|-----------|-----------|----------|--------------------|
| webnn-gpu | 1111      | 32       | 17/45              |
| webnn-npu | 1163      | 34       | 13/31              |

## Power (sudo powermetrics, 1 Hz, system-wide mW averaged over each run)

Default WebNN delegate (LiteRT):

| config        | pp t/s | tg t/s | CPU mW | GPU mW | ANE mW | ~decode mJ/token |
|---------------|--------|--------|--------|--------|--------|------------------|
| cpu           | 1059   | 537    | 7413   | 93     | **0**  | ~14              |
| webnn-default | 1213   | 33     | 8372   | 36     | **0**  | ~250             |
| webnn-cpu     | 1244   | 35     | 7822   | 0      | **0**  | ~225             |
| webnn-gpu     | 1208   | 32     | 7693   | 22     | **0**  | ~240             |
| webnn-npu     | 1221   | 33     | 7388   | 14     | **0**  | ~220             |
| webgpu        | 8734   | 301    | 5734   | 88     | **0**  | ~19              |

CoreML delegate (`WebNNCoreML,WebNNCoreMLExplicitGPUOrNPU`):

| config        | pp t/s | tg t/s | CPU mW | GPU mW | ANE mW |
|---------------|--------|--------|--------|--------|--------|
| webnn-default | 1179   | 34     | 8606   | 107    | **0**  |
| webnn-gpu     | 1192   | 32     | 8470   | 113    | **0**  |
| webnn-npu     | 1202   | 35     | 8037   | 48     | **0**  |

(CPU power is system-wide; idle floor on this machine was ~4 W during the
session, so deltas between configs are the meaningful signal. GPU power
stays near idle even for WebGPU at 77% busy — 24M-param f32 kernels are
too small to raise the GPU's DVFS state; peak GPU sample was 257 mW.)

## Findings

1. **WebGPU wins prefill by ~7.5x** over WebNN (8952 vs ~1200 t/s) and is
   the only configuration that visibly saturates the M3 GPU (75% peak).
   This is partly architectural: the WebGPU backend keeps tensors
   device-resident and compiles shaders once; the WebNN prototype pays a
   host->MLTensor->host roundtrip per op.

2. **Decode is overhead-bound everywhere at this model size.** Plain wasm
   CPU (539 t/s) beats both WebGPU (272 t/s) and WebNN (~33 t/s) at tg64:
   a 24M-param f32 decode step is so cheap that any API hop loses to a
   tight CPU loop. WebNN's ~33 t/s ≈ 90 dispatches/token x ~0.3 ms/dispatch
   — pure per-op API overhead, not compute.

3. **Prefill amortizes dispatch**: one WebNN dispatch covers all 128
   tokens, so WebNN prefill (~1200 t/s) lands slightly above the CPU
   baseline despite the roundtrips. Graph-level compilation (one MLGraph
   per cgraph) is the obvious lever and would help decode most.

4. **Chrome 149 on macOS runs WebNN on the LiteRT (TFLite) CPU delegate
   by default.** `deviceType=cpu/gpu/npu` is accepted but throughput is
   identical across all of them and GPU utilization stays at idle levels.
   With `WebNNCoreML` + `WebNNCoreMLExplicitGPUOrNPU` enabled, GPU
   activity appears (45% peak for deviceType=gpu) but throughput does not
   change — the prototype is dispatch-bound, so the compute device barely
   matters at this op granularity.

5. **Accelerator reach** — the question this was built to answer:
   WebGPU can only ever use the GPU. WebNN is the only path with a
   *mechanism* to reach the Apple Neural Engine (CoreML delegate,
   deviceType=npu), but powermetrics shows **ANE Power = 0 mW in every
   configuration measured**, including deviceType=npu with the CoreML
   delegate force-enabled. CoreML routed these graphs to CPU (plus a
   little GPU). Expected reasons: the prototype emits tiny single-op f32
   graphs, and the ANE strongly prefers f16 and only pays off for
   compiled multi-op graphs. f16 + whole-graph compilation are
   prerequisites before WebNN's NPU story can even be tested.

6. **Energy**: WebGPU is the most efficient by a wide margin — lowest
   system CPU power (5.7 W vs 7.4-8.6 W) at 8x the prefill throughput
   (~0.7 mJ/token prefill vs ~7 for CPU and WebNN). For decode, plain
   CPU is the most efficient (~14 mJ/token); per-op WebNN burns ~16x
   more energy per generated token (~230 mJ) than CPU for the same
   result — overhead, not useful work.

## Caveats

- Headless Chrome; headed runs may behave differently for GPU/CoreML.
- Tiny f32 model: decode rankings will change on larger models (WebGPU
  decode should overtake CPU well before 1B params; WebNN per-op decode
  will not).
- wasm CPU backend is single-threaded (no pthreads build).
- GPU utilization sampled at 1 Hz via ioreg over short (~10 s) runs;
  power (CPU/GPU/ANE mW) requires `sudo -v` before bench.sh.

## Additional measurements worth adding

- per-domain power + energy/token (`sudo powermetrics`) — ANE proof
- time-to-first-token and model/context load time
- pp batch-size sweep (-p 1..512) to find the dispatch-amortization knee
- larger model sweep (stories110M f32, f16 when supported)
- warm vs cold graph-compile time (WebNN MLGraph build vs WGSL pipeline
  compilation), cache-hit steady state
- numerical accuracy per delegate (LiteRT vs CoreML vs WebGPU f16 paths)
- memory: wasm heap high-water, GPU process RSS, MLTensor footprint
- thermal sustain: repeat -r 50 and watch throughput decay
