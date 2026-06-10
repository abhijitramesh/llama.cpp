#!/usr/bin/env bash
# Live demo runner for the WebGPU / WebNN / hybrid findings on Apple Silicon.
#
# Matched pairs (same model + precision, compare directly):
#   Q4_0:  ./demo.sh webgpu     vs  ./demo.sh webnn-q4   (the speed story)
#   F16:   ./demo.sh webgpu-f16 vs  ./demo.sh webnn      (the silicon story:
#          ANE is f16-only, so only the webnn f16 config lights it up)
#   ./demo.sh hybrid   phase split - ANE spike (prefill), then GPU (decode)
#   ./demo.sh power    pretty live power meter (sudo; replaces raw powermetrics)
#
# Run these in side panes first, then start a demo:
#   ./demo.sh power        (or: sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000)
#   mactop
#
# Prefill/decode are separate llama-bench tests: the pp row prints when the
# prefill phase ends, the tg row when decode ends. The workload modes also
# sample powermetrics themselves (sudo, optional) and print per-phase
# peak/average power for CPU/GPU/ANE after the rows.

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WEBNN_FLAGS="--enable-features=WebMachineLearningNeuralNetwork,WebNNCoreML,WebNNCoreMLExplicitGPUOrNPU"
PWRLOG=""

cleanup() {
    { pkill -f "http.server 910"; pkill -f "powermetrics.*-i 500";
      [ -n "${CHROME_PID:-}" ] && kill "$CHROME_PID"; wait; } 2>/dev/null
}
trap cleanup EXIT

serve() { # dir port
    # nohup + disown detaches the server from job control so killing it at
    # exit stays silent (macOS has no setsid)
    nohup python3 -m http.server "$2" --directory "$1" >/dev/null 2>&1 < /dev/null &
    disown 2>/dev/null
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

wait_for() { # pattern log -> sets WAIT_TS (epoch seconds when the row appeared)
    until grep -q "$1" "$2"; do
        sleep 1
        kill -0 "$CHROME_PID" 2>/dev/null || { echo "!!! chrome exited early, see $2"; exit 1; }
    done
    WAIT_TS=$(date +%s)
}

show_rows() { # log model-name
    # llama-bench prints the GGUF type string ("llama 256M Q4_0"); show the
    # actual model name instead
    grep -E "\| *(pp|tg)[0-9]+" "$1" \
        | sed -E 's/.*CONSOLE:?[0-9()]*\] //; s/", source.*//; s/^"//' \
        | sed -E "s/^\| [^|]+\|/| $(printf '%-31s' "${2:-model}")|/"
}

# --- per-phase power attribution -------------------------------------------
# powermetrics samples (0.5 s) are logged with epoch timestamps; each phase
# window is reconstructed backwards from the moment its result row printed,
# using the measured t/s and the known token count (x1.15 + 1 s margin).

power_start() {
    sudo -v 2>/dev/null || sudo -v || { echo "(no sudo - per-phase power disabled)"; return 0; }
    PWRLOG=$(mktemp /tmp/demo-pwr.XXXXXX)
    # -i 500 also makes these samplers distinguishable from a concurrently
    # running './demo.sh power' dashboard (-i 1000) at cleanup time
    sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 2>/dev/null | awk '
        /^CPU Power:/ { c = $3 }
        /^GPU Power:/ { g = $3 }
        /^ANE Power:/ { cmd = "date +%s"; cmd | getline t; close(cmd); print t, c, g, $3; fflush() }
    ' >"$PWRLOG" &
    disown 2>/dev/null
}

phase_power() { # label row-pattern n-tokens log t_end t_floor
    [ -n "$PWRLOG" ] && [ -s "$PWRLOG" ] || return 0
    local speed dur start
    speed=$(grep -E "\| *$2" "$4" | sed -E 's/.*\|[^0-9]*([0-9.]+) ±.*/\1/' | head -1)
    [ -n "$speed" ] || return 0
    dur=$(awk -v n="$3" -v s="$speed" 'BEGIN { printf "%d", n / s * 1.15 + 1 }')
    start=$(( $5 - dur ))
    [ "$start" -lt "$6" ] && start=$6
    awk -v s="$start" -v e="$5" -v lbl="$1" '
        $1 >= s && $1 <= e {
            if ($2 > pc) pc = $2; if ($3 > pg) pg = $3; if ($4 > pa) pa = $4;
            sc += $2; sg += $3; sa += $4; n++;
        }
        END {
            if (n) printf ">>> %-7s peak: CPU %5.2f W  GPU %5.2f W  ANE %4.0f mW    (avg %.2f / %.2f W / %.0f mW over ~%ds)\n",
                lbl, pc/1000, pg/1000, pa, sc/n/1000, sg/n/1000, sa/n, e - s;
        }' "$PWRLOG"
}

case "${1:-}" in
power)
    # animated in-place dashboard (mactop-style): redraw each sample, with
    # per-device bars (CPU 12 W / GPU 4 W / ANE 1 W full scale) and peaks.
    # ANE > 0 is the money shot for the WebNN demo.
    trap 'printf "\033[?25h\033[0m\n"' INT TERM
    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 2>/dev/null | awk '
        function bar(mw, full, color,   n, i, s) {
            n = int(mw * 30 / full); if (n > 30) n = 30;
            s = color;
            for (i = 0; i < n; i++)  s = s "\342\226\210";       # full block
            s = s "\033[2;37m";
            for (i = n; i < 30; i++) s = s "\342\226\221";       # light shade
            return s "\033[0m";
        }
        BEGIN { printf "\033[2J\033[?25l"; cpk = gpk = apk = 0 }
        /^CPU Power:/ { cpu = $3 }
        /^GPU Power:/ { gpu = $3 }
        /^ANE Power:/ {
            ane = $3;
            if (cpu > cpk) cpk = cpu;
            if (gpu > gpk) gpk = gpu;
            if (ane > apk) apk = ane;
            cmd = "date +%H:%M:%S"; cmd | getline ts; close(cmd);
            printf "\033[H";
            printf "\033[1m  Browser LLM power \342\200\224 Apple M3\033[0m              %s   \n\n", ts;
            printf "  \033[36mCPU\033[0m %7.2f W   %s  12 W   \n",  cpu/1000, bar(cpu, 12000, "\033[36m");
            printf "  \033[32mGPU\033[0m %7.2f W   %s   4 W   \n",  gpu/1000, bar(gpu,  4000, "\033[32m");
            printf "  \033[35mANE\033[0m %5.0f   mW  %s   1 W   \n\n", ane,   bar(ane,  1000, "\033[35m");
            printf "  peaks   CPU %.2f W   GPU %.2f W   ANE %.0f mW        \n", cpk/1000, gpk/1000, apk;
            fflush();
        }'
    printf "\033[?25h\033[0m\n"
    ;;
webgpu)
    LOG=/tmp/demo-webgpu.log
    serve "$ROOT/build-webgpu/bin" 9101
    power_start
    T0=$(date +%s)
    chrome_bench "" 9101 "model=stories15M-q4_0.gguf&args=-m%20/stories15M-q4_0.gguf%20-p%20128,1024%20-n%20256%20-r%204%20-ngl%2099" "$LOG"
    echo ">>> WebGPU: loading, then SHORT PREFILL pp128 (WebNN wins this length)..."
    wait_for "pp128 " "$LOG"; T_PP1=$WAIT_TS
    echo ">>> LONG PREFILL pp1024 running (WebGPU wins this one - watch GPU)..."
    wait_for "pp1024" "$LOG"; T_PP=$WAIT_TS
    echo ">>> PREFILL DONE -- DECODE running (watch GPU)"
    wait_for "tg256" "$LOG"; T_TG=$WAIT_TS
    show_rows "$LOG" "stories15M Q4_0"
    phase_power "PP128"   "pp128 " $((128 * 5))  "$LOG" "$T_PP1" "$T0"
    phase_power "PP1024"  "pp1024" $((1024 * 5)) "$LOG" "$T_PP" "$T_PP1"
    phase_power "DECODE"  "tg256"  $((256 * 5))  "$LOG" "$T_TG" "$T_PP"
    ;;
webgpu-f16)
    LOG=/tmp/demo-webgpu-f16.log
    serve "$ROOT/build-webgpu/bin" 9101
    power_start
    T0=$(date +%s)
    chrome_bench "" 9101 "model=stories15M-f16.gguf&args=-m%20/stories15M-f16.gguf%20-p%20512%20-n%20256%20-r%204%20-ngl%2099" "$LOG"
    echo ">>> WebGPU f16 (matched counterpart of './demo.sh webnn'): PREFILL..."
    wait_for "pp512" "$LOG"; T_PP=$WAIT_TS
    echo ">>> PREFILL DONE -- DECODE running (watch GPU; ANE stays 0)"
    wait_for "tg256" "$LOG"; T_TG=$WAIT_TS
    show_rows "$LOG" "stories15M F16"
    phase_power "PREFILL" "pp512" $((512 * 5)) "$LOG" "$T_PP" "$T0"
    phase_power "DECODE"  "tg256" $((256 * 5)) "$LOG" "$T_TG" "$T_PP"
    ;;
webnn)
    LOG=/tmp/demo-webnn.log
    serve "$ROOT/build-webnn/bin" 9102
    power_start
    T0=$(date +%s)
    # stories15M: 6 layers -> CoreML compiles in seconds, and the ANE signal
    # is the strongest of our models (~350 mW). Small model, same proof.
    chrome_bench "$WEBNN_FLAGS" 9102 "model=stories15M-f16.gguf&webnn=npu&f16=1&chunk=24&prune=1&args=-m%20/stories15M-f16.gguf%20-p%20512%20-n%20256%20-r%204%20-fa%201%20-ngl%2099" "$LOG"
    echo ">>> WebNN/ANE: CoreML compiling (seconds), then PREFILL (ANE engages)..."
    wait_for "pp512" "$LOG"; T_PP=$WAIT_TS
    echo ">>> PREFILL DONE -- DECODE running (~10s of sustained ANE power, watch mactop)"
    wait_for "tg256" "$LOG"; T_TG=$WAIT_TS
    show_rows "$LOG" "stories15M F16 (NPU)"
    phase_power "PREFILL" "pp512" $((512 * 5))  "$LOG" "$T_PP" "$T0"
    phase_power "DECODE"  "tg256" $((256 * 5))  "$LOG" "$T_TG" "$T_PP"
    ;;
webnn-q4)
    LOG=/tmp/demo-webnn-q4.log
    serve "$ROOT/build-webnn/bin" 9102
    power_start
    T0=$(date +%s)
    chrome_bench "$WEBNN_FLAGS" 9102 "model=stories15M-q4_0.gguf&webnn=npu&chunk=24&prune=1&args=-m%20/stories15M-q4_0.gguf%20-p%20128,1024%20-n%20256%20-r%204%20-fa%201%20-ngl%2099" "$LOG"
    echo ">>> WebNN q4: compile, then SHORT PREFILL pp128 (WebNN beats WebGPU here)..."
    wait_for "pp128 " "$LOG"; T_PP1=$WAIT_TS
    echo ">>> LONG PREFILL pp1024 running (WebGPU scales better here - honest crossover)..."
    wait_for "pp1024" "$LOG"; T_PP=$WAIT_TS
    echo ">>> PREFILL DONE -- DECODE running (int4 on CPU/GPU units, ANE stays 0)"
    wait_for "tg256" "$LOG"; T_TG=$WAIT_TS
    show_rows "$LOG" "stories15M Q4_0"
    phase_power "PP128"   "pp128 " $((128 * 5))  "$LOG" "$T_PP1" "$T0"
    phase_power "PP1024"  "pp1024" $((1024 * 5)) "$LOG" "$T_PP" "$T_PP1"
    phase_power "DECODE"  "tg256"  $((256 * 5))  "$LOG" "$T_TG" "$T_PP"
    ;;
hybrid)
    L1=/tmp/demo-hybrid-prefill.log
    L2=/tmp/demo-hybrid-decode.log
    serve "$ROOT/build-webnn/bin" 9102
    serve "$ROOT/build-webgpu/bin" 9101
    power_start
    T0=$(date +%s)
    echo ">>> HYBRID PHASE 1: PREFILL on WebNN/NPU (compile seconds, then ANE spike)"
    chrome_bench "$WEBNN_FLAGS" 9102 "model=stories15M-f16.gguf&webnn=npu&f16=1&chunk=24&prune=1&args=-m%20/stories15M-f16.gguf%20-p%20512%20-n%200%20-r%208%20-fa%201%20-ngl%2099" "$L1"
    wait_for "pp512" "$L1"; T_PP=$WAIT_TS
    kill "$CHROME_PID" 2>/dev/null
    show_rows "$L1" "stories15M F16 (NPU)"
    phase_power "PREFILL" "pp512" $((512 * 9)) "$L1" "$T_PP" "$T0"
    echo ">>> KV HANDOVER (~ms) ... PHASE 2: DECODE on WebGPU (ANE drops, GPU spikes)"
    T1=$(date +%s)
    chrome_bench "" 9101 "model=stories15M-f16.gguf&args=-m%20/stories15M-f16.gguf%20-p%200%20-n%20512%20-r%203%20-ngl%2099" "$L2"
    wait_for "tg512" "$L2"; T_TG=$WAIT_TS
    show_rows "$L2" "stories15M F16"
    phase_power "DECODE" "tg512" $((512 * 4)) "$L2" "$T_TG" "$T1"
    ;;
*)
    sed -n '2,18p' "$0"
    exit 1
    ;;
esac
