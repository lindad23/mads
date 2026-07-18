# Curriculum Baseline Reproduction Notes

This project keeps external baseline repositories and reproduced training code
separate to avoid mixing paper code with the MADS runner.

## External Sources

- ProCuRL-Target, IJCAI 2024:
  `external_baselines/procurl_target`
  Official repository: https://github.com/machine-teaching-group/ijcai2024-proximal-curriculum-target-rl
- CP-DRL:
  `external_baselines/cp_drl`
  Official repository: https://github.com/Cho-Geonwoo/CP-DRL
- DiCuRL, NeurIPS 2024:
  not integrated yet. A reliable public paper/repository matching this exact
  name was not found during the initial search; add the exact URL before
  implementation.

## Local Reproduction Layer

- `curriculum_baselines/teachers.py` contains framework-light teacher logic.
- `curriculum_baselines/adapters.py` adapts those teachers to the legacy
  `TeacherController` interface.
- `teachDeepRL/teachers/teacher_controller.py` only knows the names
  `ProCuRL-Target` and `CP-DRL`; it does not import external repositories.
- `envs/runners/adversarial_runner.py` treats `alp_gmm`, `procurl_target`, and
  `cp_drl` as parameter-space curriculum teachers using the existing
  `reset_alp_gmm` environment interface.

## Training Flags

Use `train.py` with:

```bash
python -u train.py --ued_algo=procurl_target --env_name=BipedalWalker-Adversarial-v0
python -u train.py --ued_algo=cp_drl --env_name=BipedalWalker-Adversarial-v0
```

The task vector is normalized in the legacy `[0, 2]` action convention used by
`reset_alp_gmm`; Bipedal maps it to physical terrain parameters internally.

Relevant options:

- `--procurl_beta`
- `--cp_drl_beta`
- `--cp_drl_transition_scale`
- `--cp_drl_reward_scale`
- `--cp_drl_state_scale`
- `--cp_drl_action_scale`
- `--cp_drl_aligned`
- `--cp_drl_ensemble_size`
- `--curriculum_buffer_size`
- `--curriculum_num_target_samples`
- `--curriculum_retrain_interval_episodes`

## Smoke Tests

Teacher-only smoke:

```bash
/hard_data/user/xiefeiyang/miniforge3/envs/mads38/bin/python scripts/smoke_curriculum_baselines.py
```

Training-entry smoke:

```bash
env MADS_SKIP_VIRTUAL_DISPLAY=1 MPLCONFIGDIR=/tmp/mads_mpl \
  /hard_data/user/xiefeiyang/miniforge3/envs/mads38/bin/python -u train.py \
  --xpid=smoke_procurl --ued_algo=procurl_target \
  --env_name=BipedalWalker-Adversarial-v0 \
  --log_dir=/tmp/mads_baseline_smoke --num_env_steps=4 --num_steps=4 \
  --num_processes=1 --ppo_epoch=1 --adv_ppo_epoch=1 \
  --num_mini_batch=1 --adv_num_mini_batch=1 --test_env_names= \
  --disable_checkpoint=True --checkpoint=False --no_cuda=True \
  --screenshot_interval=0 --log_interval=1 --archive_interval=0
```
