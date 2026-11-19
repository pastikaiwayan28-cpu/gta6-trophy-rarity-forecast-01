[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pastikaiwayan28-cpu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![make.fun reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pastikaiwayan28-cpu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A dilution-aware hierarchical log-time logit forecaster for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): it forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase one week after launch (2026-11-26T00:00:00Z), as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

## Model

- **Trend (median):** a robust Theil-Sen line of logit(rate) vs ln(1 + hours since release), fitted per trophy and shrunk toward its tier's mean slope (hierarchical shrinkage), read at the target.
- **Dilution:** Exophase rates fall when new players join (rate = earners / tracked). When the history is flat/stale, a dilution prior takes over: the stale tracked base is assumed to keep growing while earners grow more slowly, pulling medians down. A trophy at 100% is never forecast at 100% — new players have not earned it yet.
- **Spread (honest, asymmetric):** combines fit residual, a random-walk term growing with horizon, a player-growth term, and a flat-history term; the lower quantile is wider than the upper one, because dilution can always pull rates down.
- **Safety:** any failure in one case falls back to persistence (last rate + random-walk spread), so the output is always valid. Deterministic, no randomness, stdlib only, offline.

## Usage

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per case: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`.
