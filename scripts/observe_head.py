"""Run the head test and sample resource usage."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def read_gpu():
    """Device-wide NVIDIA metrics. Unavailable values remain None."""
    fields = (
        "memory.used,memory.total,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw"
    )
    names = (
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
        "gpu_utilization_pct",
        "gpu_memory_utilization_pct",
        "gpu_temperature_c",
        "gpu_power_w",
    )

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        values = result.stdout.strip().splitlines()[0].split(",")
        metrics = {}
        for name, value in zip(names, values):
            try:
                metrics[name] = float(value.strip())
            except ValueError:
                metrics[name] = None
        return metrics
    except (OSError, subprocess.SubprocessError, IndexError):
        return dict.fromkeys(names)


def tree_memory(pid):
    """Sum resident memory of the launched process tree."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0.0, 0

    processes = [parent]
    try:
        processes.extend(parent.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    rss = 0
    count = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return rss / (1024**2), count


def maximum(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return max(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-set", default="03")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.interval < 0.2:
        parser.error("--interval must be at least 0.2 seconds")

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "test_head.ps1"
    if not script.is_file():
        parser.error(f"Missing script: {script}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "runs" / f"observe_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", str(script),
        "-PhotoSet", args.photo_set,
    ]

    baseline_gpu = read_gpu()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    # Inherit terminal output so existing pipeline logs stay visible.
    process = subprocess.Popen(command, cwd=root)

    rows = []
    next_console = 0.0

    try:
        with (output / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = None

            while True:
                sample_started = time.perf_counter()
                elapsed = sample_started - started
                ram_mib, process_count = tree_memory(process.pid)
                system_memory = psutil.virtual_memory()

                row = {
                    "elapsed_seconds": round(elapsed, 3),
                    "process_tree_ram_mib": round(ram_mib, 2),
                    "process_count": process_count,
                    "system_ram_used_pct": system_memory.percent,
                    "system_ram_available_mib": round(
                        system_memory.available / (1024**2), 2
                    ),
                    **read_gpu(),
                }
                rows.append(row)

                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()

                writer.writerow(row)
                stream.flush()

                if elapsed >= next_console:
                    gpu_memory = row["gpu_memory_used_mib"]
                    gpu_usage = row["gpu_utilization_pct"]
                    print(
                        f"[OBS] {elapsed:.1f}s | "
                        f"RAM tree: {ram_mib:.0f} MiB | "
                        f"GPU memory: {gpu_memory} MiB | "
                        f"GPU: {gpu_usage}%",
                        flush=True,
                    )
                    next_console = elapsed + 5.0

                if process.poll() is not None:
                    break

                remaining = args.interval - (
                    time.perf_counter() - sample_started
                )
                if remaining > 0:
                    time.sleep(remaining)

        exit_code = process.wait()

    except KeyboardInterrupt:
        # Stop the launched pipeline tree if the user interrupts the monitor.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        process.wait()
        exit_code = 130

    duration = time.perf_counter() - started
    gpu_peak = maximum(rows, "gpu_memory_used_mib")
    gpu_baseline = baseline_gpu["gpu_memory_used_mib"]

    summary = {
        "status": "COMPLETED" if exit_code == 0 else "FAILED_OR_INTERRUPTED",
        "exit_code": exit_code,
        "photo_set": args.photo_set,
        "started_at_utc": started_at,
        "wall_seconds": round(duration, 3),
        "sampling_interval_seconds": args.interval,
        "sample_count": len(rows),
        "peak_process_tree_ram_mib": maximum(
            rows, "process_tree_ram_mib"
        ),
        "gpu_memory_baseline_mib": gpu_baseline,
        "peak_gpu_memory_used_mib": gpu_peak,
        "gpu_memory_peak_minus_baseline_mib": (
            round(gpu_peak - gpu_baseline, 2)
            if gpu_peak is not None and gpu_baseline is not None
            else None
        ),
        "peak_gpu_utilization_pct": maximum(
            rows, "gpu_utilization_pct"
        ),
        "peak_gpu_temperature_c": maximum(
            rows, "gpu_temperature_c"
        ),
        "peak_gpu_power_w": maximum(rows, "gpu_power_w"),
        "metrics_csv": str(output / "metrics.csv"),
        "measurement_notes": [
            "RAM is summed process-tree RSS/working set; shared pages may overlap.",
            "GPU metrics describe the whole GPU, including other applications.",
            "Peak minus baseline is not exact pipeline VRAM allocation.",
            "Sampling can miss brief peaks.",
            "Wall time includes process startup and monitoring overhead.",
        ],
    }

    report_path = output / "summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()