# Sidemen Charity Match Predictor

Interactive Python project for predicting a Sidemen Charity Match using custom lineups, user-entered player ratings, lightweight feature engineering, and Monte Carlo simulation.

## What This Repo Includes

- `sidemen_model.py`: main interactive prediction script
- `data/`: saved lineup and player-rating CSV inputs
- `outputs/`: generated prediction summaries and charts created after a run
- `apps/`, `packages/`, `services/`, `infra/`: older experimental scaffold that is not required to run the predictor

## How It Works

When you run the script, it can:

1. ask for the `SIDEMEN` lineup
2. ask for the `ALLSTARS` lineup
3. prompt you to rate each player's attributes
4. save those inputs locally
5. simulate the match and generate prediction outputs

The model combines:

- seeded historical charity match scores
- user-provided player attributes
- team feature engineering
- heuristic expected-goals estimates
- a lightweight regression blend
- Poisson-based match simulation

## Run Locally

From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn matplotlib
.venv/bin/python sidemen_model.py
```

If you already created the virtual environment, just run:

```bash
cd "/Users/ethanmoon/Documents/New project"
.venv/bin/python sidemen_model.py
```

## Outputs

After a run, the script writes files such as:

- `outputs/match_prediction_summary.json`
- `outputs/player_scorer_probabilities.csv`
- `outputs/team_feature_snapshot.csv`
- `outputs/simulation_score_distribution.csv`
- `outputs/win_probabilities.png`

## GitHub Notes

For a clean GitHub upload:

- keep `.venv/`, `node_modules/`, caches, and build artifacts out of Git
- avoid committing `outputs/` unless you want sample results in the repo
- review `data/` before pushing if it contains your personal custom ratings

## Project Status

This repository is currently centered on the Sidemen prediction script at the root. The older app and service folders are left in place, but the predictor does not depend on them.
