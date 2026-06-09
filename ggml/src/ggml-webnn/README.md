# ggml WebNN backend (experimental prototype)

Dispatches ggml ops to the browser's [WebNN API](https://www.w3.org/TR/webnn/)
(`navigator.ml`). Emscripten-only: there is no native WebNN C API, so the
backend talks to the browser through `EM_ASYNC_JS` bindings, using JSPI (or
ASYNCIFY) to suspend the wasm stack across WebNN's async calls.

**Status: feasibility prototype.** Correctness-first design:

- tensor data lives in host (wasm heap) memory, reusing the CPU buffer type
  (same architecture as the BLAS backend, device type `ACCEL`)
- `graph_compute` dispatches each ggml node as a single-op `MLGraph`,
  compiled once and cached per op signature (op + shapes + params)
- F32, contiguous tensors only

Supported ops: ADD/SUB/MUL/DIV (numpy-style broadcast), MUL_MAT (incl.
batch broadcast), SCALE, SOFT_MAX (scale + optional f32 mask), RMS_NORM,
GET_ROWS, GLU/SWIGLU (split + non-split), CPY/CONT/DUP (same-type, via
memcpy), RELU, SIGMOID, TANH, GELU_ERF, SILU, NEG, ABS, EXP, SQR, SQRT,
LOG, SIN, COS.

Known limitations / future work: no quantized or f16 tensors, no
non-contiguous tensors (attention views fall back to CPU), no ROPE /
FLASH_ATTN / SET_ROWS, per-op host<->device roundtrips (no whole-graph
compilation, no device-resident MLTensors), static-shape graphs are
rebuilt per shape (no KV-length bucketing).

## Build

```sh
emcmake cmake -B build-webnn -DGGML_WEBNN=ON -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-webnn --target test-webnn test-backend-ops llama-simple -j 8
```

`GGML_WEBNN_JSPI=ON` (default) uses JSPI; set `OFF` for ASYNCIFY.
`GGML_WEBNN_DEBUG=1` logs every dispatched graph descriptor.

## Run

WebNN needs a Chromium browser with the flag enabled
(`chrome://flags/#web-machine-learning-neural-network`, or
`--enable-features=WebMachineLearningNeuralNetwork`). Serve the build
output over HTTP and open the page; without WebNN available the backend
registers zero devices and everything falls back to CPU.

Headless run of the test suites:

```sh
cd build-webnn/bin && python3 -m http.server 8765 &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless=new --enable-features=WebMachineLearningNeuralNetwork \
    --enable-logging=stderr --user-data-dir=$(mktemp -d) \
    http://localhost:8765/test-webnn.html
```

`test-backend-ops` and `llama-simple` take CLI arguments, which an
Emscripten page passes via `Module.arguments`. A minimal runner page
(place next to the generated .js, e.g. `run-simple.html`):

```html
<script>
var Module = {
    arguments: ['-m', '/stories15M.gguf', '-n', '64', 'Once upon a time'],
    print: function(t) { console.log(t); },
    printErr: function(t) { console.error(t); },
    preRun: [function() {
        addRunDependency('fetch-model');
        fetch('stories15M.gguf').then(r => r.arrayBuffer()).then(buf => {
            FS.writeFile('/stories15M.gguf', new Uint8Array(buf));
            removeRunDependency('fetch-model');
        });
    }],
};
</script>
<script src="llama-simple.js"></script>
```

(for `test-backend-ops`, use `arguments: ['test', '-b', 'WebNN']` and no
`preRun`)

Last verified (Chrome 149 headless, macOS): `test-webnn` 31/31,
`test-backend-ops test -b WebNN` 335/335, and `llama-simple` generating
64 tokens from the f32 `tinyllamas/stories15M.gguf` with ~5.7k ops
dispatched to WebNN (`ggml-webnn: N ops were dispatched to WebNN` is
printed on backend free).

## Implementation notes

- EM_JS bodies are stringified by the C preprocessor: newlines collapse,
  so only `/* */` comments are safe inside them.
- Under JSPI, numeric/pointer arguments reach the EM_ASYNC_JS body as
  BigInt; coerce with `Number()` before `UTF8ToString` or arithmetic.
- The graph descriptor JSON doubles as the MLGraph cache key, so it must
  capture everything that changes the compiled graph.
