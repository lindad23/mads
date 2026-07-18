import argparse
import csv
import os
import time

from torch.utils.tensorboard import SummaryWriter


def _read_rows(logs_csv):
    fields = None
    rows = []
    if not os.path.exists(logs_csv):
        return fields, rows

    with open(logs_csv, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# "):
                fields = next(csv.reader([line[2:]]))
                continue
            if fields is None:
                continue
            values = next(csv.reader([line]))
            if len(values) != len(fields):
                continue
            rows.append(dict(zip(fields, values)))
    return fields, rows


def _to_float(value):
    if value in ("", "None", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def backfill_once(log_root, tags, state, writers):
    for name in sorted(os.listdir(log_root)):
        run_dir = os.path.join(log_root, name)
        logs_csv = os.path.join(run_dir, "logs.csv")
        if not os.path.isdir(run_dir) or not os.path.exists(logs_csv):
            continue

        fields, rows = _read_rows(logs_csv)
        if not fields or "steps" not in fields:
            continue

        key = run_dir
        last_step = state.get(key, -1)
        pending = []

        for row in rows:
            step = _to_float(row.get("steps"))
            if step is None:
                continue
            step = int(step)
            if step <= last_step:
                continue
            for tag in tags:
                value = _to_float(row.get(tag))
                if value is not None:
                    pending.append((tag, value, step))
            last_step = max(last_step, step)

        if pending:
            writer = writers.get(key)
            if writer is None:
                writer = SummaryWriter(os.path.join(run_dir, "tb"))
                writers[key] = writer
            for tag, value, step in pending:
                writer.add_scalar(tag, value, step)
            writer.flush()
        state[key] = last_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument(
        "--tags",
        default="mean_agent_return",
        help="Comma-separated scalar tags to mirror from logs.csv to TensorBoard.",
    )
    args = parser.parse_args()

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    state = {}
    writers = {}
    try:
        while True:
            backfill_once(args.log_root, tags, state, writers)
            time.sleep(args.interval)
    finally:
        for writer in writers.values():
            writer.close()


if __name__ == "__main__":
    main()
