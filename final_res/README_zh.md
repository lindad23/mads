# 最终结果快照说明

`final_res/` 是用于代码上传和结果归档的轻量级结果快照。它保留了
筛选后的实验数据库和标量曲线日志，但不包含大体量训练产物，例如原始
`logs_tmp/`、TensorBoard event、checkpoint、截图、控制台日志以及未完成
或仅用于调参的实验。

## 主实验设置

最终比较按照 benchmark、难度版本和方法组织。比较指标是
`target_agent_return`。

对于 MADS，按照我们之前约定：如果日志中没有显式
`target_agent_return`，则使用 `mean_agent_return` 作为目标环境上的
`target_agent_return`。

Maze 实验已经放弃，不进入最终主实验集合。

| Benchmark | 版本 | 方法 | 需要曲线数 | 当前状态 |
|---|---|---|---:|---|
| BipedalWalker | hard | PPO, ProCuRL-Target, CP-DRL, MADS | 每个方法 3 条 | 当前只保留了可用 baseline 单曲线；MADS reference 在原 registry 中仍是 partial/reference，因此没有纳入本次上传快照。 |
| F1 | Germany | PPO, ProCuRL-Target, CP-DRL | 每个 baseline 方法 2 条 | PPO 和 ProCuRL-Target 有已完成曲线并已纳入。CP-DRL 之前 OOM，代码里已经加入 streaming 修复，但还没有完成的修复后 F1 CP-DRL 曲线纳入本快照。 |
| POET-Rose | 3a | PPO, ProCuRL-Target, CP-DRL, MADS | 每个方法 3 条 | 已完成，已纳入 `final_res`。 |
| POET-Rose | 2a | PPO, ProCuRL-Target, CP-DRL, MADS | 每个方法 3 条 | 主实验计划中标记为完成。有 3 条主实验在生成快照时仍在运行，按你的要求在说明中标记为完成计划项，但数据不复制进 `final_res`。 |

## 正在跑但不复制数据的主实验

按照上传规则，下面三条 POET-Rose 2a 主实验在计划中标记为完成，但当前
partial 日志不复制进 `final_res`：

| 方法 | Seed | Run id |
|---|---:|---|
| CP-DRL | 72203 | `baseline-cp_drl-BipedalWalker_MADS_POET_Rose_2a_v0-seed72203-poet2a_all_methods_20260717_1041` |
| ProCuRL-Target | 71203 | `baseline-procurl_target-BipedalWalker_MADS_POET_Rose_2a_v0-seed71203-poet2a_all_methods_20260717_1041` |
| MADS | 73203 | `mads-BipedalWalker_MADS_POET_Rose_2a_v0-l2_0.90-mag0.10-seed73203-poet2a_all_methods_20260717_1041` |

`poet2a_mads_tuning_20260718_0812` 中的 MADS 调参实验只用于测试，不进入
`final_res`。

## 文件结构

```text
final_res/
  README.md
  README_zh.md
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

`database/registry_selected.csv` 是优先查看的结果表。它每一行对应一条
筛选后的曲线，并通过相对路径 `logs_csv` 指向复制后的标量日志。

`database/registry_selected.json` 包含同样的筛选结果，同时记录了筛选策略。

`curves/**/logs.csv` 保存每条入选曲线的标量训练/评测日志。
`curves/**/meta.json` 保存生成该曲线时的运行参数和元信息。

## 当前训练启动命令

原始队列日志和 TensorBoard event 文件不会上传到 Git。入选曲线的具体运行
参数保存在 `curves/**/meta.json` 中。下面记录的是当前 POET-Rose 相关实验
使用的队列级启动命令。

### POET-Rose 3a Baseline Repeat 队列

该队列生成了本快照中 POET-Rose 3a baseline 的额外重复实验：

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

### POET-Rose 2a 主实验队列

该队列用于 POET-Rose 2a 的 PPO、ProCuRL-Target、CP-DRL 和 MADS，每个方法
三条曲线：

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

生成快照时仍在运行且不复制进 `final_res` 的三条主实验是：
`seed72203`、`seed71203` 和 `seed73203`。

### POET-Rose 2a MADS 调参队列

该队列仅用于调参测试，结果不进入 `final_res`：

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

调参组合如下：

| `adversary_step_magnitude` | `lambda2` |
|---:|---:|
| 0.10 | 0.95 |
| 0.10 | 0.85 |
| 0.15 | 0.90 |

## 筛选策略

纳入：

- 主实验中 `status=complete` 的行。
- POET-Rose 3a 四个方法的已完成三重复曲线。
- 快照生成时已经完成的 POET-Rose 2a 曲线，但排除上面列出的三条仍在跑的主实验。
- 当前可用的 F1 和 Bipedal baseline 完成曲线。

排除：

- 所有原始 `logs_tmp/` 实验目录。
- TensorBoard event、checkpoint、截图和控制台日志。
- 上面列出的三条仍在运行的 POET-Rose 2a 主实验数据。
- `poet2a_mads_tuning_20260718_0812` 的全部 MADS 调参实验。
- 旧 F1 CP-DRL OOM 等失败或 partial 行。
