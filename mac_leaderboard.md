# MultiJudge Reliability Leaderboard

Reliability = competence (accuracy + order-consistency) gated by bias severity (position & verbosity). Higher is better. Raw accuracy is shown alongside to make the gap visible.

| Rank | Judge | Reliability | Macro-Acc | #Langs |
|---|---|---|---|---|
| 1 | Qwen2.5-7B-Instruct-4bit | 0.665 | 0.770 | 4 |

## Per-language reliability cards

| Judge | Lang | Acc | PosBiasΔ | VerbositySusc | OrderConsist | Reliability | N |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct-4bit | ar | 0.770 | -0.040 | 0.175 | 0.640 | 0.695 | 100 |
| Qwen2.5-7B-Instruct-4bit | en | 0.815 | +0.070 | 0.075 | 0.630 | 0.700 | 100 |
| Qwen2.5-7B-Instruct-4bit | hi | 0.835 | +0.090 | 0.250 | 0.670 | 0.714 | 100 |
| Qwen2.5-7B-Instruct-4bit | sw | 0.660 | -0.080 | 0.175 | 0.480 | 0.550 | 100 |