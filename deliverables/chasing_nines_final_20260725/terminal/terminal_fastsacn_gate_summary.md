# Terminal FastSACN8 promotion decision

- P7 authority promotion: **False**
- P8 five-seed promotion: **False**
- Locked state hash: `6ecb65c0721b5a3671dfe8cd6241158ce9aaed2b5a59b2fcecabdba0efe50e8f`

| Method | Variant | Seeds | Near | Task | Strict | Mean return |
|---|---|---:|---:|---:|---:|---:|
| P0 one-step | ordinary actor | 5 | 91.864% | 91.846% | 8.356% | -141.238 |
| P0 one-step | reflection + Q-search | 5 | 94.489% | 93.038% | 17.792% | -140.274 |
| P7 FastSACN8 | ordinary actor | 5 | 89.937% | 91.367% | 17.209% | -141.637 |
| P7 FastSACN8 | reflection + Q-search | 5 | 95.909% | 92.948% | 21.930% | -140.143 |

P8 is a seed-zero gate only. It cannot enter the mainline.
The authority grid was not queried by this decision.
