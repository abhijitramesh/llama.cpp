# WebGPU vs WebNN benchmark — Apple M3 MacBook, Chrome 149 (headless)

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
   deviceType=npu), but (a) it's behind extra Chrome feature flags, (b)
   ANE engagement can only be confirmed with `sudo powermetrics`
   (ANE Power counter), and (c) CoreML decides unilaterally whether to
   use ANE/GPU/CPU per graph — f32 graphs typically do NOT go to the ANE
   (it prefers f16), so f16 support in the backend is a prerequisite for
   any real ANE benefit.

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
