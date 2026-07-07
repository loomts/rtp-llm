#!/usr/bin/env python3
"""Re-align JIT workload request metrics robustly and stamp provenance.

Why this exists
---------------
The inline `request_metrics.tsv` generator in run_*.sh sorts access-log rows by
`log_time` and then zips them positionally against the workload list. Any extra or
missing access-log row (a readiness probe, a retry, a dropped line) shifts every
subsequent label silently, so group / prompt_words / ttft can all be mislabeled
without any error.

This tool instead reads the frontend "Request performance summary" log line, which
carries `request_id`, `input_len`, `reuse_len`, and the engine-side
`aux_first_token_cost_time` / `aux_cost_time`. `request_id` is a snowflake ID that
increases monotonically with global receipt order; since smoke sends the workload
strictly serially, sorting all perf rows by `request_id` reproduces send order, so we
zip them positionally against the workload list. (The line's `request_index` field is
NOT used for alignment — it is a per-frontend-server counter, and with 4 servers it
collides across servers.) `input_len == prompt_words` is emitted as an `align_ok`
cross-check column instead of silently mislabeling. It also stamps git sha / branch and
the JIT manager mode (snapshot_remote / direct_remote / direct_disabled) so a result
dir can prove which code produced it.

Output: `request_metrics_v2.tsv` next to `request_metrics.tsv`, with `#`-prefixed
provenance header lines.

Usage
-----
  reparse_metrics.py <result_dir>        # one scenario dir (main_logs/, *_jit_workloads.json, manifest.txt)
  reparse_metrics.py --run <run_root>    # every scenario subdir under a run_id root
"""
import json
import re
import subprocess
import sys
from pathlib import Path


def _grab(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def _to_num(value: str):
    if value == "":
        return ""
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except ValueError:
        return value


def _load_workloads(result_dir: Path) -> list[dict]:
    for path in sorted(result_dir.glob("*_jit_workloads.json")):
        try:
            task = json.loads(path.read_text())
        except Exception:
            continue
        items = task.get("query_result", [])
        if items:
            return [item.get("_jit_workload", {}) for item in items]
    return []


def _manifest_value(result_dir: Path, key: str) -> str:
    manifest = result_dir / "manifest.txt"
    if not manifest.is_file():
        return ""
    for line in manifest.read_text(errors="replace").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return ""


def _git(repo_root: str, *args: str) -> str:
    if not repo_root:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _detect_manager_mode(main_log_text: str) -> str:
    if "mode=direct_remote" in main_log_text:
        return "direct_remote"
    if "mode=snapshot_remote" in main_log_text:
        return "snapshot_remote"
    if "bootstrap skipped: direct remote" in main_log_text:
        return "direct_disabled"
    if "mode=local" in main_log_text:
        return "local"
    return "unknown"


def _parse_perf_summaries(main_logs: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in sorted(main_logs.glob("main_*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if "Request performance summary:" not in line:
                continue
            payload = line.split("Request performance summary:", 1)[1]
            request_id = _grab(r"request_id=(\S+)", payload)
            if not request_id:
                continue
            rows[request_id] = {
                # per-server counter, kept for reference only — NOT used for alignment
                "server_request_index": _to_num(_grab(r"request_index=(\S+)", payload)),
                "request_id": request_id,
                "completed": _grab(r"completed=(\S+)", payload),
                "input_len": _to_num(_grab(r"input_len=(\d+)", payload)),
                "output_len": _to_num(_grab(r"output_len=(\d+)", payload)),
                # engine-side timings match the old access-log columns and exclude
                # frontend queue/network, so they are the right JIT-cost signal.
                "cost_ms": _to_num(_grab(r"aux_cost_time=([0-9.]+)", payload)),
                "ttft_ms": _to_num(
                    _grab(r"aux_first_token_cost_time=([0-9.]+)", payload)
                ),
                "frontend_e2e_ms": _to_num(_grab(r"e2e_ms=([0-9.]+)", payload)),
                "frontend_ttft_ms": _to_num(
                    _grab(r"first_token_rt_ms=([0-9.]+)", payload)
                ),
                "reuse_len": _to_num(_grab(r"reuse_len=(\d+)", payload)),
            }
    parsed = list(rows.values())

    # request_id is a snowflake: numeric ascending == global receipt order == send order.
    def _rid_key(row):
        try:
            return (0, int(row["request_id"]))
        except (TypeError, ValueError):
            return (1, 0)

    parsed.sort(key=_rid_key)
    for position, row in enumerate(parsed):
        row["send_index"] = position + 1
    return parsed


COLUMNS = [
    "group",
    "workload",
    "prompt_words",
    "repeat_index",
    "max_new_tokens",
    "send_index",
    "server_request_index",
    "request_id",
    "completed",
    "input_len",
    "output_len",
    "cost_ms",
    "ttft_ms",
    "reuse_len",
    "align_ok",
]


def reparse_one(result_dir: Path) -> bool:
    main_logs = result_dir / "main_logs"
    if not main_logs.is_dir():
        print(f"skip {result_dir}: no main_logs/", file=sys.stderr)
        return False
    workloads = _load_workloads(result_dir)
    perf_rows = _parse_perf_summaries(main_logs)
    if not perf_rows:
        print(
            f"skip {result_dir}: no 'Request performance summary' lines",
            file=sys.stderr,
        )
        return False

    repo_root = _manifest_value(result_dir, "repo_root")
    main0 = main_logs / "main_0.log"
    main0_text = main0.read_text(errors="replace") if main0.is_file() else ""
    mode = _detect_manager_mode(main0_text)
    git_sha = _git(repo_root, "rev-parse", "HEAD")
    git_branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")

    mismatches = 0
    out_rows = []
    for row in perf_rows:
        idx = row["send_index"]
        wl = {}
        if isinstance(idx, int) and 1 <= idx <= len(workloads):
            wl = workloads[idx - 1]
        prompt_words = wl.get("prompt_words", "")
        align_ok = "?"
        if prompt_words != "" and row["input_len"] != "":
            align_ok = "yes" if int(prompt_words) == int(row["input_len"]) else "NO"
            if align_ok == "NO":
                mismatches += 1
        merged = {
            "group": wl.get("group", ""),
            "workload": wl.get("name", ""),
            "prompt_words": prompt_words,
            "repeat_index": wl.get("repeat_index", ""),
            "max_new_tokens": wl.get("max_new_tokens", ""),
            "align_ok": align_ok,
            **row,
        }
        out_rows.append(merged)

    out_path = result_dir / "request_metrics_v2.tsv"
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# aligned_by=request_index source=Request_performance_summary\n")
        fh.write(f"# git_sha={git_sha} git_branch={git_branch} manager_mode={mode}\n")
        fh.write(f"# repo_root={repo_root}\n")
        fh.write(
            f"# perf_rows={len(perf_rows)} workloads={len(workloads)} align_mismatches={mismatches}\n"
        )
        fh.write("\t".join(COLUMNS) + "\n")
        for row in out_rows:
            fh.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")

    flag = "" if mismatches == 0 else f"  !! {mismatches} align mismatch(es)"
    print(
        f"wrote {out_path} "
        f"(rows={len(out_rows)} workloads={len(workloads)} mode={mode} "
        f"sha={git_sha[:9]} branch={git_branch}){flag}"
    )
    return True


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "--run":
        run_root = Path(argv[1])
        found = sorted(run_root.glob("*/*/main_logs")) + sorted(
            run_root.glob("*/main_logs")
        )
        result_dirs = sorted({p.parent for p in found})
        if not result_dirs:
            print(f"no scenario dirs with main_logs/ under {run_root}", file=sys.stderr)
            return 2
        ok = all(reparse_one(d) for d in result_dirs)
        return 0 if ok else 1
    if len(argv) == 1:
        return 0 if reparse_one(Path(argv[0])) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
