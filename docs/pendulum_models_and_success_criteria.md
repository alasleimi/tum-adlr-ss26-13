# Pendulum Models And Success Criteria

This document defines the model-based references and the success criteria used in the Pendulum result pages. The goal is to avoid treating an arbitrary return threshold as an oracle.

## Environment Dynamics

All model-based calculations use the Gymnasium `Pendulum-v1` dynamics. The state is angle and angular velocity:

```text
s_t = (theta_t, theta_dot_t)
u_t in [-2, 2]
dt = 0.05
theta_dot_{t+1} = clip(theta_dot_t + (3g / (2l) * sin(theta_t) + 3 / (m l^2) * u_t) * dt, -8, 8)
theta_{t+1} = theta_t + theta_dot_{t+1} * dt
```

The per-step reward is:

```text
r_t = -(angle_normalize(theta_t)^2 + 0.1 * theta_dot_t^2 + 0.001 * u_t^2)
```

The episode horizon is 200 steps. Larger return is better, and the best return is near 0.

## Dynamic Programming Reference

The DP reference is an approximate finite-horizon planner for the Gym reward above. It solves the Bellman recursion:

```text
V_0(s) = 0
V_h(s) = max_u [ r(s, u) + V_{h-1}(f(s, u)) ]
```

where `f(s, u)` is the Gymnasium Pendulum transition model.

Approximation used for the primary report:

| Quantity | Value |
| --- | ---: |
| Horizon | `200` |
| Theta grid | `241` bins over `[-pi, pi)` |
| Angular-velocity grid | `161` bins over `[-8, 8]` |
| Torque grid | `81` bins over `[-2, 2]` |
| Value interpolation | bilinear in `(theta, theta_dot)` |
| Evaluation grid | `61 x 41` reset-support cells over `theta_dot in [-1, 1]` |

A finer check used `361 x 241` state bins and `101` torque bins. It produced the same feasible-cell counts on the 61 x 41 reset-support evaluation grid.

DP is not a mathematical proof of optimality because it discretizes states and actions. It is a near-oracle calibration for the Gym reward.

## Energy-Shaping Controller

The controller baseline is a hand-designed swing-up controller with a local PD stabilizer near upright.

Energy term:

```text
E = 0.5 * theta_dot^2 + a * cos(theta)
E* = a
a = 3g / (2l)
```

Swing-up torque:

```text
u = -k * (E - E*) * theta_dot
```

Near upright, the controller switches to local PD:

```text
u = -Kp * theta - Kd * theta_dot
```

Implementation parameters:

| Parameter | Value |
| --- | ---: |
| Energy gain `k` | `2.0` |
| `Kp` | `9.0` |
| `Kd` | `3.0` |
| Switch angle | `0.4` rad |
| Switch velocity | `3.0` |
| Torque clipping | `[-2, 2]` |

This controller is not an oracle. If it fails, the state may still be solvable by a better controller.

## Success Criteria

The primary Pendulum success criteria avoid a fixed absolute return cutoff:

| Criterion | Definition | Purpose |
| --- | --- | --- |
| Task-only success | `near_upright_fraction >= 0.8` and max not-upright streak `<= 50` | Return-independent measure of swing-up and stabilization. |
| Beats DP | `SAC_return >= DP_return` | Conservative near-oracle comparison. |
| Near DP | `SAC_return >= DP_return - 5` | Robust DP-relative comparison allowing discretization error. |
| Beats controller | `SAC_return >= controller_return` | Comparison to the hand-designed baseline. |
| Near controller | `SAC_return >= controller_return - 5` | Robust controller-relative comparison. |
| Beats best known | `SAC_return >= max(DP_return, controller_return)` | Comparison to the stronger of the two references per initial state. |
| Near best known | `SAC_return >= max(DP_return, controller_return) - 5` | Robust best-known comparison. |

The older fixed-return quantities are retained as diagnostics, not as the main success definition:

| Diagnostic | Definition | Reason to keep it |
| --- | --- | --- |
| Fixed threshold | `SAC_return >= -200` | Reproduces the original operational criterion and threshold ladder. |
| Strict threshold | Fixed threshold and task-only success | Shows how much the old threshold disagrees with actual task-state success. |

The `-200` threshold is not a principled Pendulum success definition. It is useful for continuity, but DP shows that it is not feasible from every reset-support initial state under the Gymnasium 200-step horizon and torque limit.

The `5` return-point margin is a reporting tolerance for model/discretization noise and small rollout differences. Exact "beats DP" is still reported, but it should not be the only DP-relative metric.

For heatmaps, `regret` is reported as nonnegative shortfall:

```text
regret_to_reference = max(0, reference_return - SAC_return)
signed_gap_to_reference = reference_return - SAC_return
```

Negative signed gaps mean SAC outperformed the reference on that rollout or initial-state cell. This can happen because DP is approximate and because the controller is not optimal.

## Statistical Reporting

For SAC, the primary uncertainty unit is the training seed. The reports therefore show seed-mean rates with 95% t intervals. Pooled Wilson intervals are also reported for cell-by-seed trial rates, but adjacent initial-state cells are correlated, so these are descriptive rather than independent-cell hypothesis tests.

Initial-state heatmaps show cellwise success rates across the five trained SAC seeds. They do not draw a separate confidence interval per cell. The accompanying tables and bar plots provide uncertainty summaries.
