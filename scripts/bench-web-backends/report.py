#!/usr/bin/env python3
"""Assemble a summary table from bench.sh output (bench-results/)."""

import csv
import re
import sys
from pathlib import Path

CONFIG_ORDER = ["cpu", "webnn-default", "webnn-cpu", "webnn-gpu", "webnn-npu", "webgpu"]


def parse_chrome_log(path: Path):
    """Extract llama-bench result rows (test name -> t/s) and webnn context info."""
    res = {"notes": []}
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    # console lines may be wrapped in chrome logging prefixes
    for m in re.finditer(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s*(?:\xc2\xb1|±)\s*([0-9.]+)", text):
        res[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    if "deviceType=" in text:
        m = re.search(r"created MLContext with deviceType=(\w+)", text)
        if m:
            res["notes"].append(f"MLContext deviceType={m.group(1)}")
    if re.search(r"deviceType=\w+ rejected", text):
        res["notes"].append("deviceType rejected, default context used")
    if "navigator.ml is not available" in text:
        res["notes"].append("WebNN unavailable")
    m = re.search(r"(\d+) ops were dispatched to WebNN", text)
    if m:
        res["notes"].append(f"{m.group(1)} WebNN dispatches")
    return res


def parse_samples(path: Path):
    """Average/peak GPU utilization and benchmark-Chrome CPU% over the active phase."""
    if not path.exists():
        return None
    gpu, cpu = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                if row["gpu_util_pct"]:
                    gpu.append(float(row["gpu_util_pct"]))
                cpu.append(float(row["chrome_cpu_pct"] or 0))
            except ValueError:
                continue
    out = {}
    if gpu:
        out["gpu_avg"] = sum(gpu) / len(gpu)
        out["gpu_max"] = max(gpu)
    if cpu:
        # ignore startup zeros when computing the busy average
        busy = [c for c in cpu if c > 5] or cpu
        out["cpu_avg"] = sum(busy) / len(busy)
        out["cpu_max"] = max(cpu)
    return out


def parse_power(path: Path):
    """Average mW per domain from powermetrics output."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    text = path.read_text(errors="replace")
    out = {}
    for domain in ("CPU", "GPU", "ANE"):
        vals = [float(v) for v in re.findall(rf"^{domain} Power:\s*(\d+)\s*mW", text, re.M)]
        if vals:
            out[domain] = sum(vals) / len(vals)
            out[domain + "_max"] = max(vals)
    return out


def main(outdir: str):
    out = Path(outdir)
    rows = []
    for cfg in CONFIG_ORDER:
        log = parse_chrome_log(out / f"{cfg}.chrome.log")
        if log is None:
            continue
        samples = parse_samples(out / f"{cfg}.samples.csv") or {}
        power = parse_power(out / f"{cfg}.power.txt")
        wall = (out / f"{cfg}.walltime")
        wall = wall.read_text().strip() if wall.exists() else "-"

        pp = next(((k, v) for k, v in log.items() if k.startswith("pp")), None)
        tg = next(((k, v) for k, v in log.items() if k.startswith("tg")), None)

        def fmt(item):
            return f"{item[1][0]:.2f}" if item else "-"

        row = {
            "config": cfg,
            "pp t/s": fmt(pp),
            "tg t/s": fmt(tg),
            "GPU util avg/max %": f"{samples.get('gpu_avg', 0):.0f}/{samples.get('gpu_max', 0):.0f}" if samples.get("gpu_max") is not None else "-",
            "Chrome CPU avg/max %": f"{samples.get('cpu_avg', 0):.0f}/{samples.get('cpu_max', 0):.0f}" if "cpu_max" in samples else "-",
            "CPU mW": f"{power['CPU']:.0f}" if power and "CPU" in power else "-",
            "GPU mW": f"{power['GPU']:.0f}" if power and "GPU" in power else "-",
            "ANE mW": f"{power['ANE']:.0f}" if power and "ANE" in power else "-",
            "wall s": wall,
            "notes": "; ".join(log["notes"]),
        }
        rows.append(row)

    if not rows:
        print("no results found in", outdir)
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")

    (out / "summary.md").write_text(
        "\n".join(
            ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
            + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        )
        + "\n"
    )
    print(f"\nwritten to {out}/summary.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench-results")
