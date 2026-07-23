# Specialized hybrid matched-training diagnostics

## Scope and verdict

This is a read-only, CPU-only audit of completed artifacts. It performed zero environment transitions, zero reference-policy queries, zero optimizer updates, and did not open or evaluate either authority heatmap. CPU actor inference was limited to fixed stored observations and was not used for checkpoint selection.

Artifact/budget/design verdict: **PASS**. All 50 hash-bound or freshly fingerprinted files passed; all eight endpoints are single actors with no inference router and remain below 100k optimization lineage steps.

The causal conclusions are deliberately local: saturation-aware loss changes saturated-target behavior as intended; failure mining reallocates fidelity toward failure/priority states; a 10% anchor component does not show a robust benefit in its single-seed, ten-evaluation-seed screen. None of these training diagnostics alone establishes an authority-grid improvement.

## Exact designs and budgets

| screen | completed arms | only intended contrast | optimization lineage | inclusive recorded diagnostics |
|---|---|---|---|---|
| saturation loss | 2 | action MSE vs signed-logit hinge on |target|≥1.9 | 35,000 | 36,200 (six fixed 200-step probes) |
| anchor ratio | 2 | reference_anchor_ratio 0 vs 0.1 | 35,000 | 39,000 (20 recorded 200-step eval episodes) |
| failure factorial | 4 | uniform/failure-mined × ordinary/saturation-aware | 95,000 | 95,000 (offline probes only) |

All fixed endpoints are complete. Saturation and failure arms bind checkpoints, summaries, events, progress, and metric logs through run-manifest SHA-256 values. The shared NPZs also match their manifest hashes. The anchor runner predates per-run manifests, so its configs, events, metrics, eval CSVs, replay NPZs, and final checkpoints were freshly SHA-256 fingerprinted; both event logs contain one step-5000 completion and the raw configs differ only at `sac.reference_anchor_ratio`.

## Saturation-aware loss: exact matched pair

| arm | replay MAE | replay sat MAE | replay unsat MAE | anchor MAE | anchor sat MAE | anchor unsat MAE | anchor pred sat≥.9995 | anchor mean |logit| |
|---|---|---|---|---|---|---|---|---|
| s0_action_mse_control | 0.0281869 | 0.117667 | 0.0199386 | 0.0564682 | 0.0330549 | 0.318235 | 0.29125 | 3.47224 |
| s1_saturation_aware_hybrid | 0.0338474 | 0.0932401 | 0.0283726 | 0.0553556 | 0.0236664 | 0.40965 | 0.5097 | 3.94304 |

Last-100-update telemetry:
| arm | ordinary MSE | total grad | sat grad | unsat grad | grad cosine | batch pred sat≥.9995 |
|---|---|---|---|---|---|---|
| s0_action_mse_control | 0.0065317 | 0.694231 | 0.491762 | 0.373123 | -0.0289983 | 0.0472887 |
| s1_saturation_aware_hybrid | 0.00707359 | 1.03665 | 0.826452 | 0.525667 | -0.0357365 | 0.085493 |

Already-recorded fixed hard probes (delta is final minus source H0):
| arm | probe | final return | return delta | final near | near delta | task success |
|---|---|---|---|---|---|---|
| s0_action_mse_control | neg126p885_velneg0p85 | -240.222 | -0.111194 | 0.685 | 0 | False |
| s0_action_mse_control | neg174p098_velneg0p05 | -370.287 | 4.24466 | 0.665 | 0 | False |
| s0_action_mse_control | pos126p885_velpos0p85 | -239.443 | 0.241574 | 0.705 | 0.005 | False |
| s1_saturation_aware_hybrid | neg126p885_velneg0p85 | -119.153 | 120.957 | 0.75 | 0.065 | False |
| s1_saturation_aware_hybrid | neg174p098_velneg0p05 | -331.946 | 42.5857 | 0.75 | 0.085 | False |
| s1_saturation_aware_hybrid | pos126p885_velpos0p85 | -239.74 | -0.0555315 | 0.695 | -0.005 | False |

On the replay corpus, the hinge reduces saturated-target MAE by 20.8% but increases unsaturated MAE by 42.3%. On the anchor corpus, it reduces saturated-target MAE by 28.4% while increasing unsaturated MAE by 28.7%. This is a redistribution of accuracy, not a uniform improvement.

Mechanism evidence is direct: anchor mean |logit| rises 13.6%, the ≥.9995 prediction-saturation rate rises 21.85 percentage points, and mean tanh derivative falls 19.4%. The treatment therefore pushes saturated labels farther into saturation; it does not desaturate the actor.

The logged gradient decomposition agrees. By the last 100 updates, treatment total gradient is 49% larger than control, both saturated and unsaturated components are larger, and their cosine is negative, so the strata are locally conflicting. Ordinary MSE is 8.3% worse even as the hinge objective falls. Existing fixed rollout probes improve strongly on two of three hard starts, are neutral/slightly worse on the third, and all three still fail the task criterion; these probes were fixed and reporting-only.

## Failure mining × saturation loss: full 2×2 factorial

| arm | all trajectory MAE | actual-failure MAE | failure sat MAE | nonpriority MAE | anchor MAE | anchor unsat MAE |
|---|---|---|---|---|---|---|
| d0_uniform_trajectory75_anchor25 | 0.0182375 | 0.0140413 | 0.0149279 | 0.0198606 | 0.0549792 | 0.350646 |
| d1_failure50_nonfailure25_anchor25 | 0.0186162 | 0.0130386 | 0.0137984 | 0.0205043 | 0.0557533 | 0.352158 |
| d2_uniform_trajectory75_anchor25_saturation_aware | 0.0185693 | 0.0135247 | 0.00950911 | 0.0203173 | 0.0552611 | 0.416685 |
| d3_failure50_nonfailure25_anchor25_saturation_aware | 0.018948 | 0.0120494 | 0.0067375 | 0.0209958 | 0.0564817 | 0.421989 |

Last-100-update telemetry on a common ordinary-loss scale:
| arm | ordinary SmoothL1 | total grad | sat grad | unsat grad | grad cosine |
|---|---|---|---|---|---|
| d0_uniform_trajectory75_anchor25 | 0.0108308 | 0.545984 | not logged | not logged | not logged |
| d1_failure50_nonfailure25_anchor25 | 0.00960157 | 0.523705 | not logged | not logged | not logged |
| d2_uniform_trajectory75_anchor25_saturation_aware | 0.0111534 | 0.593684 | 0.400613 | 0.42883 | -0.0613602 |
| d3_failure50_nonfailure25_anchor25_saturation_aware | 0.00997467 | 0.601352 | 0.408023 | 0.440937 | -0.0740571 |

Failure mining alone improves actual-failure MAE by 7.1% but worsens all-trajectory MAE by 2.1%, nonpriority MAE by 3.2%, and anchor MAE by 1.4%. The sampler exposes priority rows at 66.67% rather than their 20% corpus share and actual failures at about 18.85% rather than 5.67%; the tradeoff is therefore explained by intentional reweighting, not missing coverage (99.995% of trajectory rows are still seen).

The combined arm improves actual-failure MAE by 14.2% and saturated actual-failure MAE by 54.9%, but worsens all-trajectory MAE by 3.9% and anchor MAE by 2.7%. The saturated-failure interaction residual is -0.00164213 action units (negative means better than additive), evidence of useful local synergy. There is no rollout evaluation for these four endpoints, so this is mechanism evidence—not a task-performance ranking.

For both sampling modes, saturation-aware training ends with a 3–4% worse counterfactual ordinary SmoothL1 than its matched ordinary control. Its saturated and unsaturated gradients end anti-aligned (negative cosine), while saturation rates and |logit| rise. That explains why large gains on saturated failure rows coexist with worse broad-distribution fidelity.

## Ten-percent anchor during joint FastSACn + BC

| arm | recorded final mean return | task success | worst return | paired-replay-union MAE | fixed-anchor MAE | fixed-anchor unsat MAE |
|---|---|---|---|---|---|---|
| ha0_bc_onpolicy_only_anchor0 | -176.544 | 0.7 | -358.728 | 0.0268022 | 0.145977 | 0.457537 |
| ha1_bc_onpolicy90_anchor10 | -170.582 | 0.6 | -271.045 | 0.0268995 | 0.0632901 | 0.351914 |

The intervention does its intended supervised job: on the same external fixed anchor corpus, it lowers overall action MAE by 56.6%, saturated-target MAE by 68.3%, and unsaturated-target MAE by 23.1%. On the paired union of both online replay buffers, however, endpoint MAE is essentially unchanged. The fixed-anchor corpus was used only for this offline diagnostic, not selection.

The 10% anchor arm raises mean return by 5.963 but loses one task success (6/10 versus 7/10). Seven paired seeds move by less than 0.34 return points; one failure improves by 87.683, while two seeds regress materially. The mean benefit is therefore tail-dominated and is not robust evidence of better reliability.

Q-loss trajectories are effectively unchanged, as expected because the intervention enters the actor BC batch. Reference BC loss becomes more volatile, while endpoint actor/critic dormancy and effective-rank diagnostics remain very close. No gradient norms were logged for this screen, so no claim about gradient balance is made.

## Representation, movement, and gradient diagnostics

Offline activation diagnostics use identical stored rows within each matched screen and a relative dormancy threshold of 0.025. They find no widespread actor dormancy; isolated 1/64 dormant-unit readings are too small and inconsistent to explain the accuracy tradeoffs. Endpoint parameter movement is modest and similar within matched pairs. The causal signal is loss/sampling geometry, not representational collapse.

The high-resolution gradient logs exist only for the saturation-aware arms and the saturation matched pair. Ordinary failure controls log total gradient norm only; anchor-ratio arms do not log gradient norms. Missing telemetry is reported as missing rather than reconstructed post hoc.

## Scientific interpretation and next discriminating ablations

1. Keep the full 2×2 factorial as the mechanism result: mining and signed-logit saturation are locally complementary on saturated failure rows, but neither is a global fidelity win.
2. Test a capped or down-weighted signed-logit hinge (or a target logit below 4.15) with the same fixed schedules. The diagnostic target is preserving the failure-stratum gain while preventing the 19–29% unsaturated-anchor penalty.
3. Test less aggressive mining (for example 25% priority / 50% nonpriority / 25% anchor) against the existing 50/25/25 arm. The current 3.33× priority oversampling is stronger than needed for full coverage and visibly taxes nonpriority fidelity.
4. Do not promote the 10% anchor ratio on this evidence alone. Its apparent return gain is one-seed-tail dominated and task success declines. A fresh matched seed replication or a smaller anchor ratio is the necessary discriminator.
5. Any performance conclusion must come from a preregistered off-grid evaluation. The present audit intentionally provides no authority-heatmap result and makes no grid-based selection.

Implementation note: the current modules cannot express either missing dose without a small, explicit protocol extension. `failure_mined_dagger.batch_indices` and both CLIs hard-code only `uniform_control` and `failure_mined`, so a 25/50/25 dose needs a named sampling mode plus bound schedule hashes. `failure_saturation_dagger` exposes `--saturation-logit-target`, but `validate_training_numbers` pins it to 4.15; a lower target therefore also needs a versioned validation/spec change. A loss multiplier or explicit cap is not currently parameterized. These are straightforward changes, but bypassing the guards would create semantic drift and should not be done.

## Reproducible outputs

- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\diagnostics.json`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\REPORT.md`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\training_curves.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\training_curve_windows.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\offline_geometry.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\sampling_exposure.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\dormancy.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\parameter_movement.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\artifact_verification.csv`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\saturation_training_curves.png`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\failure_factorial_training_curves.png`
- `C:\Users\Ala\Desktop\Project 15\reports\systematic_joint_followup_20260722\hybrid_specialized_training_diagnostics\anchor_ratio_training_curves.png`

`training_curves.csv` contains every plotted point; `offline_geometry.csv` contains every arm/dataset/stratum metric; `sampling_exposure.csv` contains realized schedule exposure and coverage; `artifact_verification.csv` records absolute paths and SHA-256 verification; and `diagnostics.json` is the full machine-readable audit.
