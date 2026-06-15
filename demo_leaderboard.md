# MultiJudge Reliability Leaderboard

Reliability = competence (accuracy + order-consistency) gated by bias severity (position & verbosity). Higher is better. Raw accuracy is shown alongside to make the gap visible.

| Rank | Judge | Reliability | Macro-Acc | #Langs |
|---|---|---|---|---|
| 1 | fair-7b | 0.726 | 0.844 | 4 |
| 2 | verbose-7b | 0.570 | 0.750 | 4 |
| 3 | slotA-7b | 0.317 | 0.678 | 4 |

## Per-language reliability cards

| Judge | Lang | Acc | PosBiasΔ | VerbositySusc | OrderConsist | Reliability | N |
|---|---|---|---|---|---|---|---|
| fair-7b | en | 0.800 | -0.100 | 0.125 | 0.750 | 0.718 | 40 |
| fair-7b | es | 0.850 | -0.200 | 0.125 | 0.800 | 0.697 | 20 |
| fair-7b | hi | 0.875 | +0.250 | 0.000 | 0.750 | 0.660 | 20 |
| fair-7b | sw | 0.850 | +0.000 | 0.000 | 0.800 | 0.830 | 20 |
| slotA-7b | en | 0.738 | +0.525 | 0.125 | 0.475 | 0.367 | 40 |
| slotA-7b | es | 0.675 | +0.550 | 0.375 | 0.450 | 0.328 | 20 |
| slotA-7b | hi | 0.600 | +0.600 | 0.125 | 0.400 | 0.270 | 20 |
| slotA-7b | sw | 0.700 | +0.600 | 0.250 | 0.400 | 0.302 | 20 |
| verbose-7b | en | 0.725 | -0.050 | 0.625 | 0.850 | 0.614 | 40 |
| verbose-7b | es | 0.775 | +0.050 | 0.750 | 0.950 | 0.527 | 20 |
| verbose-7b | hi | 0.750 | +0.100 | 0.625 | 0.800 | 0.584 | 20 |
| verbose-7b | sw | 0.750 | +0.100 | 0.625 | 0.700 | 0.554 | 20 |