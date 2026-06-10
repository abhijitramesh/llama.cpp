#!/usr/bin/env bash
# Live demo runner for the WebGPU / WebNN / hybrid findings on Apple Silicon.
#
#   ./demo.sh webgpu   pure WebGPU  - GPU power spikes, ANE stays 0
#   ./demo.sh webnn    pure WebNN   - ANE engages (~250 mW, f16/NPU config)
#   ./demo.sh webnn-q4 pure WebNN   - the prefill speed record (q4, no ANE)
#   ./demo.sh hybrid   phase split  - ANE spike (prefill), then GPU (decode)
#   ./demo.sh power    pretty live power meter (sudo; replaces raw powermetrics)
#
# Run these in side panes first, then start a demo:
#   ./demo.sh power        (or: sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000)
#   mactop
#
# Prefill/decode are separate llama-bench tests: the pp row prints when the
# prefill phase ends, the tg row when decode ends - point at the row and the
# matching power spike together.

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WEBNN_FLAGS="--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML,WebNNCoreMLExplicitGPUOrNPU"

cleanup() { pkill -f "http.server 910" 2>/dev/null; [ -n "${CHROME_PID:-}" ] && kill "$CHROME_PID" 2>/dev/null; }
trap cleanup EXIT

serve() { # dir port
    (cd "$1" && python3 -m http.server "$2" >/dev/null 2>&1) &
    disown
    sleep 1
}

chrome_bench() { # flags port query log
    local prof
    prof=$(mktemp -d)
    # shellcheck disable=SC2086
    "$CHROME" --headless=new $1 \
        --enable-logging=stderr --v=0 --user-data-dir="$prof" --no-first-run --disable-extensions \
        "http://localhost:$2/bench.html?$3" >"$4" 2>&1 &
    CHROME_PID=$!
}

wait_for() { # pattern log
    until grep -q "$1" "$2"; do
        sleep 1
        kill -0 "$CHROME_PID" 2>/dev/null || { echo "!!! chrome exited early, see $2"; exit 1; }
    done
}

show_rows() { # log
    grep -E "\| *(pp|tg)[0-9]+" "$1" | sed -E 's/.*CONSOLE:?[0-9()]*\] //; s/", source.*//; s/^"//'
}

case "${1:-}" in
power)
    # one line per second: timestamp + watts + bars (CPU 12 W / GPU 4 W /
    # ANE 1 W full scale). ANE > 0 is the money shot for the WebNN demo.
    exec sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 2>/dev/null | awk '
        function bar(mw, full,   n, i, s) {
            n = int(mw * 24 / full); if (n > 24) n = 24;
            s = "";
            for (i = 0; i < n; i++) s = s "#";
            return sprintf("%-24s", s);
        }
        /^CPU Power:/ { cpu = $3 }
        /^GPU Power:/ { gpu = $3 }
        /^ANE Power:/ {
            ane = $3;
            cmd = "date +%H:%M:%S"; cmd | getline ts; close(cmd);
            printf "%s  CPU %6.2f W |%s|  GPU %5.2f W |%s|  ANE %5.0f mW |%s|\n",
                ts, cpu/1000, bar(cpu, 12000),
                gpu/1000, bar(gpu, 4000), ane, bar(ane, 1000);
            fflush();
        }'
    ;;
webgpu)
    LOG=/tmp/demo-webgpu.log
    serve "$ROOT/build-webgpu/bin" 9101
    chrome_bench "" 9101 "model=smollm2-135m-q4_0.gguf&args=-m%20/smollm2-135m-q4_0.gguf%20-p%201024%20-n%20256%20-r%204%20-ngl%2099" "$LOG"
    echo ">>> WebGPU: loading model, then PREFILL (watch GPU power)..."
    wait_for "pp1024" "$LOG"
    echo ">>> PREFILL DONE -- DECODE running (watch GPU)"
    wait_for "tg256" "$LOG"
    show_rows "$LOG"
    ;;
webnn)
    LOG=/tmp/demo-webnn.log
    serve "$ROOT/build-webnn/bin" 9102
    chrome_bench "$WEBNN_FLAGS" 9102 "model=smollm2-135m-f16.gguf&webnn=npu&f16=1&chunk=24&prune=1&args=-m%20/smollm2-135m-f16.gguf%20-p%201024%20-n%20128%20-r%202%20-fa%201%20-ngl%2099" "$LOG"
    echo ">>> WebNN/ANE: CoreML compiling (~30-60s), then PREFILL (ANE engages)..."
    wait_for "pp1024" "$LOG"
    echo ">>> PREFILL DONE -- DECODE running (~30s of sustained ANE power, watch mactop)"
    wait_for "tg128" "$LOG"
    show_rows "$LOG"
    ;;
webnn-q4)
    LOG=/tmp/demo-webnn-q4.log
    serve "$ROOT/build-webnn/bin" 9102
    chrome_bench "$WEBNN_FLAGS" 9102 "model=smollm2-135m-q4_0.gguf&webnn=npu&chunk=24&prune=1&args=-m%20/smollm2-135m-q4_0.gguf%20-p%201024%20-n%20256%20-r%204%20-fa%201%20-ngl%2099" "$LOG"
    echo ">>> WebNN q4 (prefill record ~1950 t/s; int4 runs on CPU/GPU units, ANE stays 0)"
    wait_for "pp1024" "$LOG"
    echo ">>> PREFILL DONE -- DECODE running"
    wait_for "tg256" "$LOG"
    show_rows "$LOG"
    ;;
hybrid)
    L1=/tmp/demo-hybrid-prefill.log
    L2=/tmp/demo-hybrid-decode.log
    serve "$ROOT/build-webnn/bin" 9102
    serve "$ROOT/build-webgpu/bin" 9101
    echo ">>> HYBRID PHASE 1: PREFILL on WebNN/NPU (compile ~30-60s, then ANE spike)"
    chrome_bench "$WEBNN_FLAGS" 9102 "model=smollm2-135m-f16.gguf&webnn=npu&f16=1&chunk=24&prune=1&args=-m%20/smollm2-135m-f16.gguf%20-p%201024%20-n%200%20-r%203%20-fa%201%20-ngl%2099" "$L1"
    wait_for "pp1024" "$L1"
    kill "$CHROME_PID" 2>/dev/null
    show_rows "$L1"
    echo ">>> KV HANDOVER (~1.7 MB, ~ms) ... PHASE 2: DECODE on WebGPU (ANE drops, GPU spikes)"
    chrome_bench "" 9101 "model=smollm2-135m-f16.gguf&args=-m%20/smollm2-135m-f16.gguf%20-p%200%20-n%20256%20-r%203%20-ngl%2099" "$L2"
    wait_for "tg256" "$L2"
    show_rows "$L2"
    ;;
*)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
