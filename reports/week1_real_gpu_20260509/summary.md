# Week 1 Extended GPU Summary

Date: 2026-05-09.

## Debug Conclusion

The original 25k Pendulum dataset was not enough. The issue was not a broken Gymnasium/DMC wrapper, a CleanRL-copy problem, or a metric logging bug. The failure mode is systematic: the SAC baseline learns an easy-start Pendulum policy but repeatedly fails the same hard evaluation starts.

## Pendulum 100k

- Runs: 5 actual seeds, 100000 steps each, CUDA.
- Evaluation: 50 fixed eval episodes at 0/25k/50k/75k/100k.
- Final return success: 150/250 = 0.60.
- Final strict success: 147/250 = 0.588.
- Final mean seed mean return: -159.4748.
- Hardest final eval seed: 100043, mean return -287.7891, strict success 0/5.

Interpretation: increasing the budget from 25k to 100k did not fix Pendulum reliability. The baseline plateaus around 60% success on the fixed eval suite.

## DMC CartPole Swingup 100k

- Runs: 3 actual seeds, 100000 steps each, CUDA.
- Evaluation: 5 fixed eval episodes at 0/25k/50k/75k/100k.
- Final return success at 850: 10/15 = 0.6667.
- Final strict success: 0/15.
- Final mean seed mean return: 855.2763.

Interpretation: DMC learns strongly by 100k, but return-threshold success and sustained-stability success disagree. Strict success remains zero because all final policies have long not-upright streaks.

## Artifacts

- Pendulum aggregate: `reports/week1_real_gpu_20260509/pendulum_100k_aggregate.json`
- Pendulum HTML: `reports/week1_real_gpu_20260509/pendulum_100k_html/index.html`
- DMC aggregate: `reports/week1_real_gpu_20260509/dmc_cartpole_100k_aggregate.json`
- DMC HTML: `reports/week1_real_gpu_20260509/dmc_cartpole_100k_html/index.html`
- Raw runs: `runs/week1_real_gpu_20260509`

## Verification

- `python -m pytest`: 16 passed.
