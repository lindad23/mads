#!/usr/bin/env python3
"""Build a small experiment-result registry for the MADS comparison runs."""

import csv
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "experiment_results" / "registry.json"
OUT_CSV = ROOT / "experiment_results" / "registry.csv"


METHODS = {
    "baseline-domain_randomization": "ppo",
    "baseline-procurl_target": "procurl_target",
    "baseline-cp_drl": "cp_drl",
}


BENCH_BY_ENV = {
    "BipedalWalker-MADS-Hard-v0": "bipedal",
    "BipedalWalker-MADS-POET-Rose-2a-v0": "poet",
    "BipedalWalker-MADS-POET-Rose-3a-v0": "poet",
    "CarRacingF1-Germany-v0": "f1",
    "CarRacingF1-MADS-Germany-v0": "f1",
}


def fnum(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def read_meta(run_dir):
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open() as fh:
        meta = json.load(fh)
    args = meta.get("args", meta)
    return args if isinstance(args, dict) else {}


def read_logs(log_path):
    if not log_path.exists():
        return [], []
    with log_path.open(newline="") as fh:
        first = fh.readline()
        if not first:
            return [], []
        if first.startswith("# "):
            header = first[2:].strip().split(",")
        elif first.startswith("#"):
            header = first[1:].strip().split(",")
        else:
            header = first.strip().split(",")
        return header, list(csv.DictReader(fh, fieldnames=header))


def infer_variant(env_name, run_dir=None, source_log_path=None):
    haystack = " ".join(
        str(value)
        for value in (env_name, run_dir, source_log_path)
        if value
    )
    if "POET-Rose-2a" in haystack or "POET_Rose_2a" in haystack:
        return "rose_2a"
    if "POET-Rose-3a" in haystack or "POET_Rose_3a" in haystack:
        return "rose_3a"
    if "BipedalWalker-MADS-Hard" in haystack or "BipedalWalker_MADS_Hard" in haystack:
        return "hard"
    if "CarRacingF1" in haystack:
        return "germany"
    return "default"


def summarize_log(log_path, metric_source):
    header, rows = read_logs(log_path)
    valid = [row for row in rows if fnum(row.get("steps")) is not None]
    if not valid:
        return {
            "num_points": 0,
            "last_logged_step": None,
            "max_logged_step": None,
            "final_value": None,
            "best_value": None,
            "best_step": None,
        }

    def metric(row):
        if metric_source == "mean_agent_return":
            return fnum(row.get("mean_agent_return"))
        value = fnum(row.get("target_agent_return"))
        if value is not None:
            return value
        for key in header:
            if key.startswith("test_returns:"):
                value = fnum(row.get(key))
                if value is not None:
                    return value
        return fnum(row.get("mean_agent_return"))

    points = []
    for row in valid:
        step = fnum(row.get("steps"))
        value = metric(row)
        if value is not None:
            points.append((step, value))

    if not points:
        best_step = best_value = final_value = None
    else:
        best_step, best_value = max(points, key=lambda item: item[1])
        final_value = points[-1][1]

    return {
        "num_points": len(valid),
        "last_logged_step": fnum(valid[-1].get("steps")),
        "max_logged_step": max(fnum(row.get("steps")) for row in valid),
        "final_value": final_value,
        "best_value": best_value,
        "best_step": best_step,
    }


def status_for(summary, total_steps, force_status=None):
    if force_status:
        return force_status
    max_step = summary["max_logged_step"]
    if total_steps and max_step and max_step >= 0.98 * total_steps:
        return "complete"
    if max_step:
        return "partial"
    return "missing"


def add_entry(entries, *, bench, method, curve_id, run_dir, metric_source,
              total_steps=None, status=None, source_log_path=None, notes="",
              variant=None):
    run_dir = Path(run_dir)
    log_path = run_dir / "logs.csv"
    meta = read_meta(run_dir)
    if total_steps is None:
        total_steps = fnum(meta.get("num_env_steps"))
    env_name = meta.get("env_name")
    summary = summarize_log(log_path, metric_source)
    entries.append({
        "bench": bench,
        "variant": variant or infer_variant(env_name, run_dir, source_log_path),
        "method": method,
        "curve_id": curve_id,
        "status": status_for(summary, total_steps, status),
        "metric_name": "target_agent_return",
        "metric_source": metric_source,
        "run_dir": str(run_dir.resolve()),
        "logs_csv": str(log_path.resolve()),
        "source_log_path": str(Path(source_log_path).resolve()) if source_log_path else None,
        "xpid": meta.get("xpid", run_dir.name),
        "env_name": env_name,
        "seed": meta.get("seed"),
        "total_steps": total_steps,
        **summary,
        "notes": notes,
    })


def baseline_entries(entries):
    root = ROOT / "logs_tmp" / "baselines_dense_target_eval_20260714_2045"
    for log_path in sorted(root.glob("baseline-*/logs.csv")):
        run_dir = log_path.parent
        name = run_dir.name
        method = next(
            (value for prefix, value in METHODS.items() if name.startswith(prefix)),
            None,
        )
        if method is None:
            continue
        meta = read_meta(run_dir)
        env_name = meta.get("env_name")
        bench = BENCH_BY_ENV.get(env_name)
        if bench is None:
            continue
        force = None
        if name.startswith("baseline-cp_drl-CarRacingF1_MADS_Germany"):
            force = "running_resume"
        add_entry(
            entries,
            bench=bench,
            method=method,
            curve_id="seed%s" % meta.get("seed", "unknown"),
            run_dir=run_dir,
            metric_source="target_agent_return",
            status=force,
        )


def mads_entries(entries):
    board_root = ROOT / "logs_tmp" / "baselines_dense_target_eval_20260714_2045"
    add_entry(
        entries,
        bench="bipedal",
        method="mads",
        curve_id="dps095_reference_resume_needed",
        run_dir=board_root / "mads-reference-BipedalWalker_MADS_Hard_v0-best_mean_as_target-20260715_170821",
        metric_source="target_agent_return",
        total_steps=50_000_000,
        status="partial_needs_resume",
        source_log_path=ROOT / "logs_tmp" / "four_runs_dps095_20260703_131118" / "BipedalWalker-MADS-Hard-v0-dps095-gpu3-20260703_131118" / "logs.csv",
        notes="Current Bipedal MADS reference. target_agent_return is copied from mean_agent_return; needs resume to 50M.",
    )

    poet_runs = [
        (
            "l2_090_mag010_grid_best_final",
            board_root / "mads-reference-BipedalWalker_MADS_POET_Rose_3a_v0-l2_090_mag010_mean_as_target-20260715_170821",
            ROOT / "logs_tmp" / "poet_rose_hard_mag_l2_grid_20260704_233900" / "BipedalWalker-MADS-POET-Rose-3a-v0-dps-l2-0.90-mag0.10-gpu3-20260704_233900" / "logs.csv",
            "target_agent_return",
        ),
        (
            "l2_095_mag010_rep5",
            ROOT / "logs_tmp" / "poet_rose_hard_replicates_l2_095_mag010_20260706_022807" / "BipedalWalker-MADS-POET-Rose-3a-v0-dps-l2-0.95-mag0.10-rep5-gpu7-20260706_022807",
            None,
            "mean_agent_return",
        ),
        (
            "l2_090_mag010_sweep",
            ROOT / "logs_tmp" / "poet_rose_hard_mag_sweep_l2_090_20260705_124907" / "BipedalWalker-MADS-POET-Rose-3a-v0-dps-l2-0.90-mag0.10-gpu2-20260705_124907",
            None,
            "mean_agent_return",
        ),
    ]
    for curve_id, run_dir, source_log, metric_source in poet_runs:
        add_entry(
            entries,
            bench="poet",
            method="mads",
            curve_id=curve_id,
            run_dir=run_dir,
            metric_source=metric_source,
            source_log_path=source_log,
            total_steps=50_000_000,
            notes="Selected among the top POET-Rose 3a MADS curves by final mean_agent_return.",
        )

    for log_path in sorted(board_root.glob("mads-BipedalWalker_MADS_POET_Rose_2a_v0-*/logs.csv")):
        run_dir = log_path.parent
        meta = read_meta(run_dir)
        add_entry(
            entries,
            bench="poet",
            method="mads",
            curve_id="seed%s" % meta.get("seed", run_dir.name),
            run_dir=run_dir,
            metric_source="mean_agent_return",
            total_steps=fnum(meta.get("num_env_steps")) or 50_000_000,
            notes="POET-Rose 2a MADS repeat. target_agent_return is reported from mean_agent_return for MADS.",
            variant="rose_2a",
        )


def build_missing_plan(entries):
    registered_counts = {}
    complete_counts = {}
    for entry in entries:
        if entry["status"] != "missing":
            key = (entry["bench"], entry["variant"], entry["method"])
            registered_counts[key] = registered_counts.get(key, 0) + 1
        if entry["status"] == "complete":
            key = (entry["bench"], entry["variant"], entry["method"])
            complete_counts[key] = complete_counts.get(key, 0) + 1

    desired = {}
    for method in ("ppo", "procurl_target", "cp_drl", "mads"):
        desired[("bipedal", "hard", method)] = 3
    for method in ("ppo", "procurl_target", "cp_drl"):
        desired[("f1", "germany", method)] = 2
    desired[("f1", "germany", "mads")] = 0
    for method in ("ppo", "procurl_target", "cp_drl"):
        desired[("poet", "rose_3a", method)] = 3
    desired[("poet", "rose_3a", "mads")] = 3
    for method in ("ppo", "procurl_target", "cp_drl", "mads"):
        desired[("poet", "rose_2a", method)] = 3

    plan = []
    for bench, variant, method in sorted(desired):
        have = registered_counts.get((bench, variant, method), 0)
        complete = complete_counts.get((bench, variant, method), 0)
        want = desired[(bench, variant, method)]
        missing = max(0, want - have)
        action = "none"
        if missing:
            action = "schedule_new_runs"
        elif complete < want:
            action = "wait_for_registered_runs_to_finish"
        if bench == "bipedal" and variant == "hard" and method == "mads":
            action = "resume_current_to_50M_then_schedule_new_runs"
        if bench == "f1" and variant == "germany" and method == "cp_drl":
            action = "wait_for_current_resume_then_schedule_one_more_run"
        if bench == "f1" and variant == "germany" and method == "mads":
            action = "skip_for_now"
        plan.append({
            "bench": bench,
            "variant": variant,
            "method": method,
            "desired_curves": want,
            "registered_curves": have,
            "complete_curves": complete,
            "missing_curves": missing,
            "action": action,
        })
    return plan


def write_csv(entries):
    fields = [
        "bench", "variant", "method", "curve_id", "status", "metric_name",
        "metric_source", "final_value", "best_value", "best_step",
        "last_logged_step", "max_logged_step", "total_steps",
        "run_dir", "logs_csv", "source_log_path", "xpid", "env_name",
        "seed", "notes",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry.get(key) for key in fields})


def main():
    entries = []
    baseline_entries(entries)
    mads_entries(entries)
    entries.sort(key=lambda item: (item["bench"], item["method"], item["curve_id"]))
    data = {
        "schema_version": 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "metric_policy": {
            "baseline": "target_agent_return if present; otherwise test_returns:<target_env>.",
            "mads": "target_agent_return is copied from mean_agent_return when a board reference exists; otherwise metric_source=mean_agent_return.",
        },
        "desired_layout": {
            "benches": ["f1", "bipedal", "poet"],
            "variants": {
                "bipedal": ["hard"],
                "f1": ["germany"],
                "poet": ["rose_2a", "rose_3a"],
            },
            "methods": ["ppo", "procurl_target", "cp_drl", "mads"],
            "replicates": {
                "bipedal": 3,
                "f1_baselines": 2,
                "f1_mads": 0,
                "poet_rose_2a_all_methods": 3,
                "poet_rose_3a_mads": 3,
                "poet_rose_3a_baselines": 3,
            },
        },
        "entries": entries,
        "missing_plan": build_missing_plan(entries),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    write_csv(entries)
    print(OUT_JSON)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
