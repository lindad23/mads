#!/usr/bin/env bash
set -u

REPO_ROOT="/hard_data/user/xiefeiyang/coding/MADS-main"
PYTHON="/hard_data/user/xiefeiyang/miniforge3/envs/mads38/bin/python"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs_tmp/baselines_hard_queue_20260707_${STAMP}}"

# Use all eight physical GPUs as candidates, but only launch on GPUs that look
# actually idle. The queue will never place two baseline jobs on the same GPU.
GPU_POOL=(${GPU_POOL:-0 1 2 3 4 5 6 7})
MAX_PARALLEL="${MAX_PARALLEL:-6}"
IDLE_MEM_MB="${IDLE_MEM_MB:-3000}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-15}"
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

MAZE_ARGS=(
  --gamma=0.995
  --lr=0.0001
  --num_steps=256
  --num_processes=32
  --num_env_steps="${MAZE_NUM_ENV_STEPS:-50000000}"
  --ppo_epoch=5
  --num_mini_batch=1
  --entropy_coef=0.0
  --adv_entropy_coef=0.0
  --recurrent_arch=lstm
  --recurrent_agent=True
  --recurrent_adversary_env=False
  --recurrent_hidden_size=256
  --test_interval=25
  --test_num_episodes=10
  --test_num_processes=2
  --log_interval=25
  --checkpoint_basis=student_grad_updates
  --archive_interval=5000
  --log_action_complexity=True
  --log_replay_complexity=True
  --log_plr_buffer_stats=True
)

BIPEDAL_ARGS=(
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
  --test_interval="${BIPEDAL_TEST_INTERVAL:-5}"
  --test_num_episodes="${BIPEDAL_TEST_NUM_EPISODES:-5}"
  --test_num_processes="${BIPEDAL_TEST_NUM_PROCESSES:-2}"
  --log_interval="${BIPEDAL_LOG_INTERVAL:-5}"
  --checkpoint_basis=student_grad_updates
  --log_replay_complexity=True
  --log_plr_buffer_stats=True
)

F1_ARGS=(
  --num_steps=125
  --num_processes=16
  --num_env_steps=5500000
  --ppo_epoch=8
  --num_mini_batch=4
  --entropy_coef=0.0
  --adv_entropy_coef=0.01
  --num_action_repeat=8
  --frame_stack=4
  --grayscale=False
  --crop_frame=False
  --reward_shaping=True
  --test_interval="${F1_TEST_INTERVAL:-25}"
  --test_num_episodes="${F1_TEST_NUM_EPISODES:-3}"
  --test_num_processes="${F1_TEST_NUM_PROCESSES:-3}"
  --log_interval="${F1_LOG_INTERVAL:-5}"
  --checkpoint_basis=student_grad_updates
  --archive_interval=1250
  --cp_drl_model_batch_size="${F1_CP_DRL_MODEL_BATCH_SIZE:-128}"
  --log_action_complexity=False
  --log_plr_buffer_stats=True
)

JOBS=(
  "ppo_maze|domain_randomization|MultiGrid-Task2-v0|MultiGrid-Task2-v0|7001|maze"
  "ppo_bipedal|domain_randomization|BipedalWalker-MADS-Hard-v0|BipedalWalker-MADS-Hard-Eval-v0|7002|bipedal"
  "ppo_poetrose|domain_randomization|BipedalWalker-MADS-POET-Rose-3a-v0|BipedalWalker-MADS-POET-Rose-3a-Eval-v0|7003|bipedal"
  "ppo_f1_germany|domain_randomization|CarRacingF1-Germany-v0|CarRacingF1-Germany-v0|7004|f1"
  "procurl_maze|procurl_target|MultiGrid-Task2-v0|MultiGrid-Task2-v0|7101|maze"
  "procurl_bipedal|procurl_target|BipedalWalker-MADS-Hard-v0|BipedalWalker-MADS-Hard-Eval-v0|7102|bipedal"
  "procurl_poetrose|procurl_target|BipedalWalker-MADS-POET-Rose-3a-v0|BipedalWalker-MADS-POET-Rose-3a-Eval-v0|7103|bipedal"
  "procurl_f1_germany|procurl_target|CarRacingF1-MADS-Germany-v0|CarRacingF1-Germany-v0|7104|f1"
  "cpdrl_maze|cp_drl|MultiGrid-Task2-v0|MultiGrid-Task2-v0|7201|maze"
  "cpdrl_bipedal|cp_drl|BipedalWalker-MADS-Hard-v0|BipedalWalker-MADS-Hard-Eval-v0|7202|bipedal"
  "cpdrl_poetrose|cp_drl|BipedalWalker-MADS-POET-Rose-3a-v0|BipedalWalker-MADS-POET-Rose-3a-Eval-v0|7203|bipedal"
  "cpdrl_f1_germany|cp_drl|CarRacingF1-MADS-Germany-v0|CarRacingF1-Germany-v0|7204|f1"
)

declare -A ACTIVE_PIDS=()
declare -A ACTIVE_RUNS=()
declare -A ACTIVE_STARTS=()
declare -a PENDING=("${JOBS[@]}")

if [[ -n "${JOB_FILTER:-}" ]]; then
  IFS=',' read -r -a FILTER_LABELS <<< "${JOB_FILTER}"
  declare -A WANT_LABEL=()
  for label in "${FILTER_LABELS[@]}"; do
    WANT_LABEL["${label}"]=1
  done

  FILTERED=()
  for job in "${PENDING[@]}"; do
    IFS='|' read -r label _ <<< "${job}"
    if [[ -n "${WANT_LABEL[$label]+set}" ]]; then
      FILTERED+=("${job}")
    fi
  done
  PENDING=("${FILTERED[@]}")
fi

mkdir -p "${LOG_ROOT}"
STATUS_FILE="${LOG_ROOT}/queue_status.tsv"
RUNS_FILE="${LOG_ROOT}/runs.csv"
if [[ "${APPEND_QUEUE_LOGS:-0}" == "1" ]]; then
  touch "${STATUS_FILE}" "${RUNS_FILE}"
else
  : > "${STATUS_FILE}"
  : > "${RUNS_FILE}"
fi

cd "${REPO_ROOT}" || exit 1

log_status() {
  printf '%s\t%s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS_FILE}"
}

gpu_is_active() {
  local gpu="$1"
  [[ -n "${ACTIVE_PIDS[$gpu]+set}" ]]
}

gpu_is_idle() {
  local gpu="$1"
  local line mem util
  line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1)"
  [[ -n "${line}" ]] || return 1
  IFS=',' read -r _ mem util <<< "${line}"
  mem="${mem// /}"
  util="${util// /}"
  [[ "${mem}" =~ ^[0-9]+$ ]] || return 1
  [[ "${util}" =~ ^[0-9]+$ ]] || return 1
  [[ "${mem}" -le "${IDLE_MEM_MB}" && "${util}" -le "${IDLE_UTIL_PCT}" ]]
}

active_count() {
  local count=0
  local gpu
  for gpu in "${!ACTIVE_PIDS[@]}"; do
    if kill -0 "${ACTIVE_PIDS[$gpu]}" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

reap_finished() {
  local gpu pid code run started ended
  for gpu in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$gpu]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      run="${ACTIVE_RUNS[$gpu]}"
      started="${ACTIVE_STARTS[$gpu]}"
      ended="$(date '+%F %T')"
      if wait "${pid}"; then
        code=0
      else
        code=$?
      fi
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
  local label ued env test_env seed kind xpid run_dir
  IFS='|' read -r label ued env test_env seed kind <<< "${job}"
  local effective_test_env="${test_env}"
  # MultiGrid eval currently hits an observation-shape mismatch with the
  # recurrent baseline model. Keep training alive and evaluate maze separately.
  if [[ "${kind}" == "maze" ]]; then
    effective_test_env=""
  fi

  xpid="baseline-${ued}-${env//-/_}-seed${seed}-${STAMP}"
  run_dir="${LOG_ROOT}/${xpid}"
  mkdir -p "${run_dir}"
  printf '%s,%s,%s,%s,%s,%s\n' "${label}" "${gpu}" "${ued}" "${env}" "${test_env}" "${xpid}" >> "${RUNS_FILE}"

  local -a kind_args=()
  case "${kind}" in
    maze) kind_args=("${MAZE_ARGS[@]}") ;;
    bipedal) kind_args=("${BIPEDAL_ARGS[@]}") ;;
    f1) kind_args=("${F1_ARGS[@]}") ;;
    *) log_status "unknown_kind kind=${kind} job=${job}"; return 1 ;;
  esac

  log_status "starting gpu=${gpu} label=${label} ued=${ued} env=${env} xpid=${xpid}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MADS_SKIP_VIRTUAL_DISPLAY=1
    if [[ "${kind}" == "f1" || "${env}" == CarRacing* ]]; then
      export DISPLAY="${MADS_F1_DISPLAY:-:99}"
    fi
    export MPLCONFIGDIR="/tmp/mads_mpl_${USER}_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"
    "${PYTHON}" -u train.py \
      --xpid="${xpid}" \
      --ued_algo="${ued}" \
      --env_name="${env}" \
      --test_env_names="${effective_test_env}" \
      --seed="${seed}" \
      --log_dir="${LOG_ROOT}" \
      "${COMMON_ARGS[@]}" \
      "${kind_args[@]}"
  ) > "${run_dir}/console.log" 2>&1 &

  ACTIVE_PIDS["${gpu}"]=$!
  ACTIVE_RUNS["${gpu}"]="${label}"
  ACTIVE_STARTS["${gpu}"]="$(date '+%F %T')"
}

log_status "queue_start log_root=${LOG_ROOT} max_parallel=${MAX_PARALLEL} idle_mem_mb=${IDLE_MEM_MB} idle_util_pct=${IDLE_UTIL_PCT} gpu_pool=${GPU_POOL[*]} job_filter=${JOB_FILTER:-all}"

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
