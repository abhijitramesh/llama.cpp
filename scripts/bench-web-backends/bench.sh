#!/usr/bin/env bash
# Benchmark llama.cpp wasm backends (CPU / WebNN / WebGPU) in Chrome on macOS.
#
# Usage:
#   scripts/bench-web-backends/bench.sh [config ...]
#
# Configs (default: all):
#   cpu webnn-default webnn-cpu webnn-gpu webnn-npu webgpu
#
# Expects:
#   build-webnn/bin/llama-bench.{js,wasm}   (emcmake build with -DGGML_WEBNN=ON)
#   build-webgpu/bin/llama-bench.{js,wasm}  (emcmake build with -DGGML_WEBGPU=ON)
#   stories15M.gguf (or $MODEL) in both bin dirs
#
# Host-side sampling per run (1 Hz):
#   - GPU utilization %        (ioreg, no privileges needed)
#   - benchmark Chrome CPU %   (ps over the run's own --user-data-dir processes)
#   - CPU/GPU/ANE power in mW  (powermetrics, only if sudo credentials are cached:
#                               run `sudo -v` first to enable; this is the only way
#                               to observe Apple Neural Engine activity)
#
# Results land in bench-results/: <config>.chrome.log, <config>.samples.csv,
# <config>.power.txt, then `python3 report.py` assembles a summary table.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
MODEL="${MODEL:-stories15M.gguf}"
BENCH_ARGS="${BENCH_ARGS:--m /$MODEL -p 128 -n 64 -r 3}"
PORT="${PORT:-8799}"
TIMEOUT_S="${TIMEOUT_S:-600}"
OUT="$ROOT/bench-results"
WEBNN_FLAG="${WEBNN_FLAG:---enable-features=WebMachineLearningNeuralNetwork}"

mkdir -p "$OUT"

have_powermetrics=0
if sudo -n true 2>/dev/null; then
    have_powermetrics=1
    echo "powermetrics: enabled (sudo cached) - CPU/GPU/ANE power will be recorded"
else
    echo "powermetrics: DISABLED (run 'sudo -v' before this script to record CPU/GPU/ANE power)"
fi

sample_loop() { # $1 = csv out, $2 = user-data-dir to attribute CPU%
    echo "ts,gpu_util_pct,chrome_cpu_pct" > "$1"
    while :; do
        g=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -oE '"Device Utilization %"=[0-9]+' | head -1 | grep -oE '[0-9]+$')
        pids=$(pgrep -f "$2" 2>/dev/null)
        c=0
        if [ -n "$pids" ]; then
            c=$(ps -o %cpu= -p $(echo "$pids" | tr '\n' ',' | sed 's/,$//') 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s+0}')
        fi
        echo "$(date +%s),${g:-},${c:-0}" >> "$1"
        sleep 1
    done
}

run_config() { # $1 = name, $2 = bin dir, $3 = extra query, $4 = extra chrome flags
    local name=$1 bindir=$2 query=$3 flags=$4

    if [ ! -f "$bindir/llama-bench.js" ]; then
        echo "[$name] SKIP: $bindir/llama-bench.js not found"
        return
    fi
    if [ ! -f "$bindir/$MODEL" ]; then
        echo "[$name] SKIP: $bindir/$MODEL not found"
        return
    fi

    cp "$SCRIPT_DIR/bench.html" "$bindir/bench.html"

    local log="$OUT/$name.chrome.log"
    local prof; prof=$(mktemp -d "/tmp/benchprof.$name.XXXXXX")

    (cd "$bindir" && exec python3 -m http.server "$PORT") >/dev/null 2>&1 &
    local srv_pid=$!
    sleep 1

    sample_loop "$OUT/$name.samples.csv" "$prof" &
    local sampler_pid=$!
    disown $sampler_pid 2>/dev/null

    local pm_pid=
    if [ "$have_powermetrics" = 1 ]; then
        sudo -n powermetrics -i 1000 --samplers cpu_power,gpu_power,ane_power \
            > "$OUT/$name.power.txt" 2>/dev/null &
        pm_pid=$!
    fi

    local args_enc; args_enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$BENCH_ARGS")
    local url="http://localhost:$PORT/bench.html?model=$MODEL&args=$args_enc$query"

    echo "[$name] $url"
    local t0; t0=$(date +%s)
    "$CHROME" --headless=new $flags \
        --enable-logging=stderr --v=0 \
        --user-data-dir="$prof" --no-first-run --disable-extensions \
        "$url" > "$log" 2>&1 &
    local chrome_pid=$!

    # llama-bench's final row is the tg test; the wasm runtime stays alive after
    # main() returns (EXIT_RUNTIME=0), so completion is detected from the output
    local done_pat="BENCH_EXIT_CODE|\|[ ]*tg[0-9]+[ ]*\|"
    local waited=0
    while ! grep -qE "$done_pat" "$log" 2>/dev/null; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge "$TIMEOUT_S" ]; then
            echo "[$name] TIMEOUT after ${TIMEOUT_S}s"
            break
        fi
        if ! kill -0 $chrome_pid 2>/dev/null; then
            echo "[$name] chrome exited early"
            break
        fi
    done
    sleep 3 # let trailing output flush
    local t1; t1=$(date +%s)
    echo "$((t1 - t0))" > "$OUT/$name.walltime"

    kill $chrome_pid 2>/dev/null
    kill $sampler_pid 2>/dev/null
    [ -n "$pm_pid" ] && { sudo -n pkill -f "powermetrics -i 1000" 2>/dev/null; kill $pm_pid 2>/dev/null; }
    kill $srv_pid 2>/dev/null
    wait $chrome_pid $sampler_pid $srv_pid 2>/dev/null
    rm -rf "$prof"

    grep -E "BENCH_EXIT_CODE|\| *(pp|tg)[0-9]+" "$log" | sed -E 's/.*CONSOLE:?[0-9()]*\] //; s/", source.*//; s/^"//' | sed "s/^/[$name] /"
    echo "[$name] done in $((t1 - t0))s"
}

CONFIGS="${*:-cpu webnn-default webnn-cpu webnn-gpu webnn-npu webgpu}"

for cfg in $CONFIGS; do
    case "$cfg" in
        cpu)           run_config cpu           "$ROOT/build-webnn/bin"  ""             "" ;;
        webnn-default) run_config webnn-default "$ROOT/build-webnn/bin"  ""             "$WEBNN_FLAG" ;;
        webnn-cpu)     run_config webnn-cpu     "$ROOT/build-webnn/bin"  "&webnn=cpu"   "$WEBNN_FLAG" ;;
        webnn-gpu)     run_config webnn-gpu     "$ROOT/build-webnn/bin"  "&webnn=gpu"   "$WEBNN_FLAG" ;;
        webnn-npu)     run_config webnn-npu     "$ROOT/build-webnn/bin"  "&webnn=npu"   "$WEBNN_FLAG" ;;
        webgpu)        BENCH_ARGS="$BENCH_ARGS -ngl 99" run_config webgpu "$ROOT/build-webgpu/bin" "" "" ;;
        *) echo "unknown config: $cfg" ;;
    esac
done

echo
python3 "$SCRIPT_DIR/report.py" "$OUT"
