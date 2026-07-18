# Final Result Snapshot

This folder is the lightweight result snapshot for code release. It keeps the
experiment database and selected scalar logs, while excluding large training
artifacts, TensorBoard event files, checkpoints, screenshots, and unfinished
or tuning-only runs.

## Main Experiments

The main comparison is organized by benchmark, difficulty variant, and method.
The comparison metric is `target_agent_return`.

For MADS runs, `target_agent_return` follows our agreed convention: it is the
fixed-target agent performance, represented by `mean_agent_return` when the
MADS log does not contain an explicit `target_agent_return` field.

Maze experiments are intentionally abandoned and are not part of the final
main experiment set.

| Benchmark | Variant | Methods | Required curves | Current status |
|---|---|---|---:|---|
| BipedalWalker | hard | PPO, ProCuRL-Target, CP-DRL, MADS | 3 per method | Baseline single curves are available; MADS reference is not included in this upload snapshot because the registry marks it as partial/reference data. |
| F1 | Germany | PPO, ProCuRL-Target, CP-DRL | 2 per baseline method | PPO and ProCuRL-Target have completed selected curves. CP-DRL previously OOMed; the streaming CP-DRL fix is in code, but no completed fixed F1 CP-DRL curve is included yet. |
| POET-Rose | 3a | PPO, ProCuRL-Target, CP-DRL, MADS | 3 per method | Completed and included in `final_res`. |
| POET-Rose | 2a | PPO, ProCuRL-Target, CP-DRL, MADS | 3 per method | Marked complete for the main experiment plan. Three runs were still active when this snapshot was created and are documented as complete-by-plan, but their data is not copied into `final_res`. |

## Running Main Runs Not Copied

Per the release rule, the following POET-Rose 2a main runs are marked as
completed in the experiment plan, but their current partial logs are not copied
into `final_res`:

| Method | Seed | Run id |
|---|---:|---|
| CP-DRL | 72203 | `baseline-cp_drl-BipedalWalker_MADS_POET_Rose_2a_v0-seed72203-poet2a_all_methods_20260717_1041` |
| ProCuRL-Target | 71203 | `baseline-procurl_target-BipedalWalker_MADS_POET_Rose_2a_v0-seed71203-poet2a_all_methods_20260717_1041` |
| MADS | 73203 | `mads-BipedalWalker_MADS_POET_Rose_2a_v0-l2_0.90-mag0.10-seed73203-poet2a_all_methods_20260717_1041` |

MADS tuning runs from `poet2a_mads_tuning_20260718_0812` are test-only and are
also excluded from this folder.

## Folder Structure

```text
final_res/
  README.md
  database/
    registry_selected.csv
    registry_selected.json
  curves/
    <bench>/
      <variant>/
        <method>/
          <curve_id>/
            logs.csv
            meta.json
```

`database/registry_selected.csv` is the table to use first. It contains one row
per selected curve and points to the copied scalar log through the relative
`logs_csv` path.

`database/registry_selected.json` contains the same selected entries plus the
selection policy.

`curves/**/logs.csv` contains the scalar training/evaluation curve for each
selected run. `curves/**/meta.json` stores the command-line arguments and run
metadata used to produce that curve.

## Launch Commands

The raw queue logs and TensorBoard event files are intentionally ignored by Git.
The exact run arguments for selected curves are preserved in
`curves/**/meta.json`. The commands below record the queue-level commands used
for the currently launched POET-Rose runs.

### POET-Rose 3a Baseline Repeats

This queue produced the extra POET-Rose 3a baseline repeats included in this
snapshot:

```bash
tmux new-session -d -s poet_baseline_repeats_20260716 \
  "cd /hard_data/user/xiefeiyang/coding/MADS-main && \
   STAMP=poet_repeats_20260716_1110 \
   GPU_POOL='5 3 2 7' \
   MAX_PARALLEL=4 \
   IDLE_MEM_MB=12000 \
   IDLE_UTIL_PCT=85 \
   POLL_SECONDS=60 \
   ./scripts/run_poet_baseline_repeats_20260716.sh"
```

### POET-Rose 2a Main Queue

This is the main POET-Rose 2a queue for PPO, ProCuRL-Target, CP-DRL, and MADS,
three curves per method:

```bash
tmux new-session -d -s poet2a_all_methods_20260717 \
  "cd /hard_data/user/xiefeiyang/coding/MADS-main && \
   STAMP=poet2a_all_methods_20260717_1041 \
   GPU_POOL='4 5 2 3 7 0 1 6' \
   MAX_PARALLEL=6 \
   IDLE_MEM_MB=10000 \
   IDLE_UTIL_PCT=100 \
   POLL_SECONDS=60 \
   ./scripts/run_poet2a_all_methods_repeats_20260717.sh"
```

The following three main POET-Rose 2a runs were still active when this snapshot
was made and are therefore documented in the plan but not copied into
`final_res`: `seed72203`, `seed71203`, and `seed73203`.

### POET-Rose 2a MADS Tuning Queue

This queue is tuning-only and is intentionally excluded from `final_res`:

```bash
tmux new-session -d -s poet2a_mads_tuning_20260718 \
  "cd /hard_data/user/xiefeiyang/coding/MADS-main && \
   STAMP=poet2a_mads_tuning_20260718_0812 \
   GPU_POOL='4 3 2 7 5 0 1 6' \
   MAX_PARALLEL=3 \
   IDLE_MEM_MB=8000 \
   IDLE_UTIL_PCT=100 \
   POLL_SECONDS=60 \
   ./scripts/run_poet2a_mads_tuning_20260718.sh"
```

The tuning parameter combinations were:

| `adversary_step_magnitude` | `lambda2` |
|---:|---:|
| 0.10 | 0.95 |
| 0.10 | 0.85 |
| 0.15 | 0.90 |

## Selection Policy

Included:

- Main experiment rows with `status=complete`.
- Completed POET-Rose 3a curves for all four methods.
- Completed POET-Rose 2a curves available at snapshot time, excluding the three
  active main runs listed above.
- Completed available F1 and Bipedal baseline curves.

Excluded:

- All raw `logs_tmp/` experiment directories.
- TensorBoard event files, checkpoints, screenshots, and console logs.
- The three active POET-Rose 2a main runs listed above.
- All MADS tuning runs from `poet2a_mads_tuning_20260718_0812`.
- Failed or partial rows such as the old F1 CP-DRL OOM run.
