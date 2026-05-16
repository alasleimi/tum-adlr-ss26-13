# CleanRL SAC Audit

Status: no implementation bug found that explains the Pendulum reliability gap.

## What Is Actually CleanRL

- The actor and critic classes are imported directly from the copied CleanRL file: `cleanrl.sac_continuous_action.Actor` and `SoftQNetwork`.
- The replay buffer implementation is copied from CleanRL's Stable-Baselines3 dependency path under `cleanrl_utils.buffers`.
- The Week 1 code wraps those pieces for telemetry, fixed-seed evaluation, replay inspection, checkpointing, and DMC/Gymnasium observation flattening. It does not reimplement SAC network math.

## Configuration Equivalence

The 100k Pendulum baseline uses the CleanRL SAC defaults for the SAC hyperparameters that matter most:

- hidden architecture: 2 layers, width 256, exactly as copied CleanRL
- actor learning rate: `3e-4`
- critic learning rate and alpha learning rate: `1e-3`
- gamma: `0.99`
- tau: `0.005`
- learning starts: `5000`
- batch size: `256`
- policy frequency: `2`
- target network frequency: `1`
- automatic entropy target: `-action_dim`

Intentional differences:

- `env_id` is `Pendulum-v1` rather than CleanRL's Hopper default.
- `total_steps` is an experiment-budget variable.
- `buffer_size=100000` for the 100k baseline stores every collected transition. For longer runs the investigation script uses `buffer_size=500000`.
- We use a single non-vector Gymnasium env. CleanRL's script also defaults to one environment.
- We evaluate deterministic policies at fixed seeds and log more diagnostics; this does not change training.

## Update Loop Check

The critic target, critic loss, actor loss, entropy coefficient update, and target-network soft update match the copied CleanRL code.

Truncations are handled consistently with CleanRL's intent. CleanRL bootstraps through time-limit truncation by using termination flags rather than treating every timeout as terminal. Our non-vector loop stores `done=terminated`, not `terminated or truncated`, so Pendulum time limits are also bootstrapped.

The only exact-equivalence caveat is the actor-update counter. CleanRL gates actor updates on `global_step % policy_frequency == 0`; our wrapper gates on optimizer `update_step % policy_frequency == 0`. For the CleanRL baseline setting `learning_starts=5000`, `updates_per_step=1`, `policy_frequency=2`, these have the same parity and therefore the same update cadence. For `updates_per_step > 1`, the behavior is a deliberate scale variant and should not be labeled "exact CleanRL baseline."

## Reliability Diagnosis

The low Pendulum reliability is not explained by an obvious CleanRL config or terminal-state bug. The evidence points to a real hard-start failure mode:

- 1000-episode post-hoc eval from the five 100k checkpoints gives mean seed fixed-threshold diagnostic `0.7012` and strict-threshold diagnostic `0.6734`.
- The dense reset-support map over Gymnasium's actual initial velocity range `[-1, 1]` gives cell mean fixed-threshold diagnostic `0.6918` and strict-threshold diagnostic `0.6692`.
- Failures concentrate near downward starts, especially around `theta = +/-174 to +/-180 degrees` with small or opposing angular velocity.
- High-velocity states across the full Pendulum state range are much easier, so full-state success maps must not be confused with the reset-distribution reliability.

## Remaining Risks

- The project still needs an oracle or near-oracle Pendulum calibration. Current thresholds are operational success criteria, not proof of optimality.
- UTD and longer-budget runs are experimental variants, not exact CleanRL. They are needed to determine whether the failure mode is caused by too little environment interaction, too few optimizer updates, replay distribution, or policy-class limitations.
- Pooled episode Wilson intervals reuse fixed eval seeds across training seeds. They are useful for episode-level failure rates, but seed means are the safer unit for cross-run claims.
