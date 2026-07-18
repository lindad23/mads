#!/usr/bin/env bash
set -u

REPO_ROOT="/hard_data/user/xiefeiyang/coding/MADS-main"
PYTHON="/hard_data/user/xiefeiyang/miniforge3/envs/mads38/bin/python"
STAMP="${STAMP:-poet2a_mads_tuning_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs_tmp/baselines_dense_target_eval_20260714_2045}"
STATUS_FILE="${STATUS_FILE:-${LOG_ROOT}/poet2a_mads_tuning_status_${STAMP}.tsv}"
RUNS_FILE="${RUNS_FILE:-${LOG_ROOT}/poet2a_mads_tuning_runs_${STAMP}.csv}"

GPU_POOL=(${GPU_POOL:-4 3 2 7 5 0 1 6})
MAX_PARALLEL="${MAX_PARALLEL:-3}"
IDLE_MEM_MB="${IDLE_MEM_MB:-8000}"
IDLE_UTIL_PCT="${IDLE_UTIL_PCT:-100}"
POLL_SECONDS="${POLL_SECONDS:-60}"

ENV_NAME="BipedalWalker-MADS-POET-Rose-2a-v0"
TEST_ENV_NAME="BipedalWalker-MADS-POET-Rose-2a-Eval-v0"
ENV_TOKEN="BipedalWalker_MADS_POET_Rose_2a_v0"

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
  "mads_poet2a_mag010_l2_095|0.10|0.95|74201"
  "mads_poet2a_mag010_l2_085|0.10|0.85|74202"
  "mads_poet2a_mag015_l2_090|0.15|0.90|74203"
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
      (cd "${REPO_ROOT}" && python scripts/build_experiment_registry.py >/dev/null 2>&1 || true)
    fi
  done
}

launch_job() {
  local gpu="$1"
  local job="$2"
  local label mag lambda2 seed mag_token l2_token xpid run_dir
  IFS='|' read -r label mag lambda2 seed <<< "${job}"
  mag_token="$(printf '%s' "${mag}" | tr -d '.')"
  l2_token="$(printf '%s' "${lambda2}" | tr -d '.')"
  xpid="mads-${ENV_TOKEN}-tune-mag${mag_token}-l2_${l2_token}-seed${seed}-${STAMP}"
  run_dir="${LOG_ROOT}/${xpid}"
  mkdir -p "${run_dir}"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "${label}" "${gpu}" "${mag}" "${lambda2}" "${ENV_NAME}" "${TEST_ENV_NAME}" "${seed}" "${xpid}" >> "${RUNS_FILE}"
  log_status "starting gpu=${gpu} label=${label} mag=${mag} lambda2=${lambda2} env=${ENV_NAME} xpid=${xpid}"
  (
    cd "${REPO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export MADS_SKIP_VIRTUAL_DISPLAY=1
    export MPLCONFIGDIR="/tmp/mads_mpl_${USER}_${gpu}"
    mkdir -p "${MPLCONFIGDIR}"
    "${PYTHON}" -u train_mads.py \
      --xpid="${xpid}" \
      --env_name="${ENV_NAME}" \
      --test_env_names="${TEST_ENV_NAME}" \
      --seed="${seed}" \
      --log_dir="${LOG_ROOT}" \
      "${COMMON_ARGS[@]}" \
      "${POET_ARGS[@]}" \
      --ued_algo=paired \
      --lambda1=0.95 \
      --lambda2="${lambda2}" \
      --adversary_step_magnitude="${mag}"
  ) > "${run_dir}/console.log" 2>&1 &

  ACTIVE_PIDS["${gpu}"]=$!
  ACTIVE_RUNS["${gpu}"]="${label}"
  ACTIVE_STARTS["${gpu}"]="$(date '+%F %T')"
}

printf 'label,gpu,adversary_step_magnitude,lambda2,env,test_env,seed,xpid\n' > "${RUNS_FILE}"
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

(cd "${REPO_ROOT}" && python scripts/build_experiment_registry.py >/dev/null 2>&1 || true)
log_status "queue_done"
