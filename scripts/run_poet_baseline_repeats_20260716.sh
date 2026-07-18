#!/usr/bin/env bash
set -u

REPO_ROOT="/hard_data/user/xiefeiyang/coding/MADS-main"
PYTHON="/hard_data/user/xiefeiyang/miniforge3/envs/mads38/bin/python"
STAMP="${STAMP:-poet_repeats_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs_tmp/baselines_dense_target_eval_20260714_2045}"
STATUS_FILE="${STATUS_FILE:-${LOG_ROOT}/poet_repeats_status_${STAMP}.tsv}"
RUNS_FILE="${RUNS_FILE:-${LOG_ROOT}/poet_repeats_runs_${STAMP}.csv}"

GPU_POOL=(${GPU_POOL:-5 3 2 7})
MAX_PARALLEL="${MAX_PARALLEL:-4}"
IDLE_MEM_MB="${IDLE_MEM_MB:-12000}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-80}"
POLL_SECONDS="${POLL_SECONDS:-60}"

COMMON_ARGS=(
  --use_gae=True
  --gamma=0.99
  --gae_lambda=0.9
  --lr=0.0003
  --max_grad_norm=0.5
  --algo=ppo
  --clip_param=0.2
  --value_loss_coef=0.5
  --clip_value_loss=False
  --normalize_returns=True
  --handle_timelimits=True
  --checkpoint=True
  --disable_checkpoint=False
  --log_grad_norm=True
  --screenshot_interval=0
  --archive_interval=5000
  --cuda_device=0
)

POET_ARGS=(
  --num_steps=2048
  --num_processes=16
  --num_env_steps=50000000
  --ppo_epoch=5
  --num_mini_batch=32
  --entropy_coef=0.001
  --adv_entropy_coef=0.01
  --adv_ppo_epoch=8
  --adv_num_mini_batch=4
  --adv_normalize_returns=True
  --adv_use_popart=False
  --test_interval="${POET_TEST_INTERVAL:-5}"
  --test_num_episodes="${POET_TEST_NUM_EPISODES:-5}"
  --test_num_processes="${POET_TEST_NUM_PROCESSES:-2}"
  --log_interval="${POET_LOG_INTERVAL:-5}"
  --checkpoint_basis=student_grad_updates
  --log_replay_complexity=True
  --log_plr_buffer_stats=True
)

JOBS=(
  "ppo_poetrose_rep2|domain_randomization|70013"
  "ppo_poetrose_rep3|domain_randomization|70023"
  "procurl_poetrose_rep2|procurl_target|71013"
  "procurl_poetrose_rep3|procurl_target|71023"
  "cpdrl_poetrose_rep2|cp_drl|72013"
  "cpdrl_poetrose_rep3|cp_drl|72023"
)

declare -A ACTIVE_PIDS=()
declare -A ACTIVE_RUNS=()
declare -A ACTIVE_STARTS=()
declare -a PENDING=("${JOBS[@]}")

mkdir -p "${LOG_ROOT}"

log_status() {
  printf '%s\t%s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS_FILE}"
}

active_count() {
  echo "${#ACTIVE_PIDS[@]}"
}

gpu_is_active() {
  local gpu="$1"
  [[ -n "${ACTIVE_PIDS[$gpu]:-}" ]]
}

gpu_is_idle() {
  local gpu="$1"
  local row used free util
  row="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F ', ' -v g="${gpu}" '$1 == g {print $0}')"
  [[ -n "${row}" ]] || return 1
  IFS=', ' read -r _ used free util <<< "${row}"
  [[ "${free}" -ge "${IDLE_MEM_MB}" && "${util}" -le "${IDLE_UTIL_PCT}" ]]
}

reap_finished() {
  local gpu pid run started code ended
  for gpu in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$gpu]}"
    run="${ACTIVE_RUNS[$gpu]}"
    started="${ACTIVE_STARTS[$gpu]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"
      code=$?
      ended="$(date '+%F %T')"
      log_status "finished gpu=${gpu} pid=${pid} exit=${code} run=${run} started=${started} ended=${ended}"
      unset ACTIVE_PIDS["${gpu}"]
      unset ACTIVE_RUNS["${gpu}"]
      unset ACTIVE_STARTS["${gpu}"]
    fi
  done
}

launch_job() {
  local gpu="$1"
  local job="$2"
  local label ued seed env test_env xpid run_dir
  IFS='|' read -r label ued seed <<< "${job}"
  env="BipedalWalker-MADS-POET-Rose-3a-v0"
  test_env="BipedalWalker-MADS-POET-Rose-3a-Eval-v0"
  xpid="baseline-${ued}-BipedalWalker_MADS_POET_Rose_3a_v0-seed${seed}-${STAMP}"
  run_dir="${LOG_ROOT}/${xpid}"
  mkdir -p "${run_dir}"
  printf '%s,%s,%s,%s,%s,%s\n' "${label}" "${gpu}" "${ued}" "${env}" "${test_env}" "${xpid}" >> "${RUNS_FILE}"
  log_status "starting gpu=${gpu} label=${label} ued=${ued} env=${env} xpid=${xpid}"
  (
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MADS_SKIP_VIRTUAL_DISPLAY=1
    export MPLCONFIGDIR="/tmp/mads_mpl_${USER}_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"
    "${PYTHON}" -u train.py \
      --xpid="${xpid}" \
      --ued_algo="${ued}" \
      --env_name="${env}" \
      --test_env_names="${test_env}" \
      --seed="${seed}" \
      --log_dir="${LOG_ROOT}" \
      "${COMMON_ARGS[@]}" \
      "${POET_ARGS[@]}"
  ) > "${run_dir}/console.log" 2>&1 &

  ACTIVE_PIDS["${gpu}"]=$!
  ACTIVE_RUNS["${gpu}"]="${label}"
  ACTIVE_STARTS["${gpu}"]="$(date '+%F %T')"
}

log_status "queue_start log_root=${LOG_ROOT} max_parallel=${MAX_PARALLEL} idle_mem_mb=${IDLE_MEM_MB} idle_util_pct=${IDLE_UTIL_PCT} gpu_pool=${GPU_POOL[*]} stamp=${STAMP}"

while [[ "${#PENDING[@]}" -gt 0 || "$(active_count)" -gt 0 ]]; do
  reap_finished

  if [[ "${#PENDING[@]}" -gt 0 && "$(active_count)" -lt "${MAX_PARALLEL}" ]]; then
    for gpu in "${GPU_POOL[@]}"; do
      [[ "${#PENDING[@]}" -gt 0 ]] || break
      [[ "$(active_count)" -lt "${MAX_PARALLEL}" ]] || break
      gpu_is_active "${gpu}" && continue
      if gpu_is_idle "${gpu}"; then
        next_job="${PENDING[0]}"
        PENDING=("${PENDING[@]:1}")
        launch_job "${gpu}" "${next_job}"
      fi
    done
  fi

  if [[ "${#PENDING[@]}" -gt 0 || "$(active_count)" -gt 0 ]]; then
    log_status "heartbeat pending=${#PENDING[@]} active=$(active_count)"
    sleep "${POLL_SECONDS}"
  fi
done

log_status "queue_done"
