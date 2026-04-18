#!/usr/bin/env python3
"""
Sidemen Charity Match Prediction Model
--------------------------------------
A portfolio-style end-to-end pipeline for predicting a Sidemen Charity Match.

What this script does:
1. Creates local data folders and CSV templates if they do not exist
2. Seeds a small public historical match dataset
3. Lets you manually rate players and specify lineups
4. Engineers player-level and team-level features
5. Trains a lightweight score model when enough data exists
6. Falls back to a strong heuristic model when data is sparse
7. Simulates the match thousands of times with a Poisson engine
8. Saves outputs, probabilities, and charts

Dependencies:
    pip install pandas numpy scikit-learn matplotlib

Optional:
    pip install xgboost

How to use:
1. Run once:
       python sidemen_model.py
   This creates:
       data/historical_matches.csv
       data/player_ratings.csv
       data/current_lineups.csv

2. Open data/player_ratings.csv
   Fill in better ratings for each player.

3. Open data/current_lineups.csv
   Mark which players are on SIDEMEN or ALLSTARS for the target match.

4. Run again:
       python sidemen_model.py

Outputs:
    outputs/match_prediction_summary.json
    outputs/simulation_score_distribution.csv
    outputs/team_feature_snapshot.csv
    outputs/win_probabilities.png
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MPL_CACHE_DIR = BASE_DIR / ".mplcache"

HISTORICAL_MATCHES_CSV = DATA_DIR / "historical_matches.csv"
PLAYER_RATINGS_CSV = DATA_DIR / "player_ratings.csv"
CURRENT_LINEUPS_CSV = DATA_DIR / "current_lineups.csv"

RANDOM_SEED = 42
SIMULATIONS = 20000

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(RANDOM_SEED)

# You can change this target match year if you want
TARGET_MATCH_LABEL = "Target Match"

# -----------------------------
# Seed data
# -----------------------------

# Public historical match outcomes used as seed data.
# You can expand this later with more detailed player-level match logs.
#
# Notes:
# - 2016: Sidemen FC 7-2 YouTube Allstars
# - 2017: Sidemen FC 2-0 YouTube Allstars
# - 2018: Sidemen FC 7-1 YouTube Allstars
# - 2022: Sidemen FC 8-7 YouTube Allstars
# - 2023: Sidemen FC 8-5 YouTube Allstars
# - 2025: Sidemen FC 9-9 YouTube Allstars (Allstars won on penalties, but regulation was a draw)
#
# We model regulation/full-time goals only.
SEEDED_HISTORICAL_MATCHES = [
    {"year": 2016, "sidemen_goals": 7, "allstars_goals": 2},
    {"year": 2017, "sidemen_goals": 2, "allstars_goals": 0},
    {"year": 2018, "sidemen_goals": 7, "allstars_goals": 1},
    {"year": 2022, "sidemen_goals": 8, "allstars_goals": 7},
    {"year": 2023, "sidemen_goals": 8, "allstars_goals": 5},
    {"year": 2025, "sidemen_goals": 9, "allstars_goals": 9},
]

# Seed player ratings.
# These are intentionally rough priors. You should edit them manually later.
# Scale suggestion:
#   0-100 for skill-like attributes
#   seriousness: how hard they are likely to try
#   volatility: how unpredictable they are
#   expected_minutes_default: 0-90
#
# These names are examples / priors only.
SEEDED_PLAYER_RATINGS = [
    # Sidemen / common returners
    {"player": "Miniminter", "attack": 87, "defense": 55, "passing": 76, "fitness": 74, "goalkeeping": 5, "seriousness": 88, "volatility": 32, "experience": 95, "chemistry_core": 95, "expected_minutes_default": 80, "preferred_position": "FWD"},
    {"player": "TBJZL", "attack": 81, "defense": 64, "passing": 74, "fitness": 82, "goalkeeping": 5, "seriousness": 86, "volatility": 35, "experience": 94, "chemistry_core": 95, "expected_minutes_default": 80, "preferred_position": "MID"},
    {"player": "Behzinga", "attack": 67, "defense": 72, "passing": 68, "fitness": 76, "goalkeeping": 5, "seriousness": 87, "volatility": 38, "experience": 93, "chemistry_core": 95, "expected_minutes_default": 78, "preferred_position": "DEF"},
    {"player": "Vikkstar123", "attack": 63, "defense": 61, "passing": 71, "fitness": 68, "goalkeeping": 5, "seriousness": 80, "volatility": 42, "experience": 93, "chemistry_core": 95, "expected_minutes_default": 75, "preferred_position": "MID"},
    {"player": "Zerkaa", "attack": 66, "defense": 64, "passing": 67, "fitness": 70, "goalkeeping": 5, "seriousness": 82, "volatility": 40, "experience": 93, "chemistry_core": 95, "expected_minutes_default": 76, "preferred_position": "MID"},
    {"player": "KSI", "attack": 58, "defense": 56, "passing": 54, "fitness": 61, "goalkeeping": 62, "seriousness": 85, "volatility": 55, "experience": 92, "chemistry_core": 95, "expected_minutes_default": 70, "preferred_position": "GK"},
    {"player": "W2S", "attack": 72, "defense": 46, "passing": 58, "fitness": 57, "goalkeeping": 5, "seriousness": 70, "volatility": 60, "experience": 92, "chemistry_core": 95, "expected_minutes_default": 65, "preferred_position": "FWD"},

    # Common YouTube charity-match players
    {"player": "Manny", "attack": 84, "defense": 62, "passing": 77, "fitness": 84, "goalkeeping": 5, "seriousness": 89, "volatility": 30, "experience": 94, "chemistry_core": 82, "expected_minutes_default": 82, "preferred_position": "FWD"},
    {"player": "ChrisMD", "attack": 82, "defense": 58, "passing": 75, "fitness": 79, "goalkeeping": 5, "seriousness": 88, "volatility": 34, "experience": 94, "chemistry_core": 84, "expected_minutes_default": 82, "preferred_position": "FWD"},
    {"player": "Theo Baker", "attack": 78, "defense": 57, "passing": 74, "fitness": 78, "goalkeeping": 5, "seriousness": 84, "volatility": 36, "experience": 78, "chemistry_core": 78, "expected_minutes_default": 75, "preferred_position": "FWD"},
    {"player": "Chunkz", "attack": 60, "defense": 50, "passing": 62, "fitness": 56, "goalkeeping": 5, "seriousness": 72, "volatility": 58, "experience": 72, "chemistry_core": 72, "expected_minutes_default": 60, "preferred_position": "MID"},
    {"player": "Angryginge", "attack": 61, "defense": 70, "passing": 64, "fitness": 68, "goalkeeping": 5, "seriousness": 86, "volatility": 44, "experience": 55, "chemistry_core": 68, "expected_minutes_default": 72, "preferred_position": "DEF"},
    {"player": "IShowSpeed", "attack": 69, "defense": 45, "passing": 50, "fitness": 80, "goalkeeping": 5, "seriousness": 83, "volatility": 72, "experience": 65, "chemistry_core": 60, "expected_minutes_default": 66, "preferred_position": "FWD"},
    {"player": "JME", "attack": 54, "defense": 49, "passing": 58, "fitness": 52, "goalkeeping": 5, "seriousness": 70, "volatility": 46, "experience": 60, "chemistry_core": 66, "expected_minutes_default": 55, "preferred_position": "MID"},
    {"player": "Max Fosh", "attack": 48, "defense": 46, "passing": 57, "fitness": 54, "goalkeeping": 5, "seriousness": 64, "volatility": 67, "experience": 52, "chemistry_core": 64, "expected_minutes_default": 50, "preferred_position": "MID"},
    {"player": "WillNE", "attack": 57, "defense": 49, "passing": 58, "fitness": 60, "goalkeeping": 5, "seriousness": 73, "volatility": 49, "experience": 72, "chemistry_core": 70, "expected_minutes_default": 60, "preferred_position": "MID"},
    {"player": "Jynxzi", "attack": 52, "defense": 44, "passing": 50, "fitness": 58, "goalkeeping": 5, "seriousness": 69, "volatility": 63, "experience": 20, "chemistry_core": 38, "expected_minutes_default": 45, "preferred_position": "FWD"},
    {"player": "Logan Paul", "attack": 74, "defense": 60, "passing": 63, "fitness": 85, "goalkeeping": 5, "seriousness": 78, "volatility": 53, "experience": 28, "chemistry_core": 35, "expected_minutes_default": 55, "preferred_position": "FWD"},
    {"player": "Mark Rober", "attack": 35, "defense": 38, "passing": 44, "fitness": 42, "goalkeeping": 5, "seriousness": 72, "volatility": 40, "experience": 10, "chemistry_core": 18, "expected_minutes_default": 25, "preferred_position": "MID"},
    {"player": "Kai Cenat", "attack": 56, "defense": 40, "passing": 47, "fitness": 69, "goalkeeping": 5, "seriousness": 66, "volatility": 61, "experience": 35, "chemistry_core": 45, "expected_minutes_default": 40, "preferred_position": "FWD"},
    {"player": "xQc", "attack": 42, "defense": 36, "passing": 40, "fitness": 41, "goalkeeping": 5, "seriousness": 58, "volatility": 73, "experience": 22, "chemistry_core": 28, "expected_minutes_default": 30, "preferred_position": "MID"},
]

# A starter lineup template.
# Edit this file after first run.
SEEDED_CURRENT_LINEUPS = [
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "Miniminter", "expected_minutes": 80, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "TBJZL", "expected_minutes": 78, "position": "MID", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "Behzinga", "expected_minutes": 76, "position": "DEF", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "Vikkstar123", "expected_minutes": 72, "position": "MID", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "Zerkaa", "expected_minutes": 74, "position": "MID", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "KSI", "expected_minutes": 60, "position": "GK", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "SIDEMEN", "player": "W2S", "expected_minutes": 58, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "Manny", "expected_minutes": 82, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "ChrisMD", "expected_minutes": 82, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "Theo Baker", "expected_minutes": 74, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "Chunkz", "expected_minutes": 58, "position": "MID", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "Angryginge", "expected_minutes": 74, "position": "DEF", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "IShowSpeed", "expected_minutes": 62, "position": "FWD", "is_starter": 1},
    {"match_label": TARGET_MATCH_LABEL, "team": "ALLSTARS", "player": "JME", "expected_minutes": 55, "position": "MID", "is_starter": 1},
]

# -----------------------------
# Utility functions
# -----------------------------


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def create_seed_files_if_missing() -> None:
    if not HISTORICAL_MATCHES_CSV.exists():
        pd.DataFrame(SEEDED_HISTORICAL_MATCHES).to_csv(HISTORICAL_MATCHES_CSV, index=False)
        print(f"Created {HISTORICAL_MATCHES_CSV}")

    if not PLAYER_RATINGS_CSV.exists():
        pd.DataFrame(SEEDED_PLAYER_RATINGS).to_csv(PLAYER_RATINGS_CSV, index=False)
        print(f"Created {PLAYER_RATINGS_CSV}")

    if not CURRENT_LINEUPS_CSV.exists():
        pd.DataFrame(SEEDED_CURRENT_LINEUPS).to_csv(CURRENT_LINEUPS_CSV, index=False)
        print(f"Created {CURRENT_LINEUPS_CSV}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    while True:
        response = input(f"{prompt}{suffix}: ").strip()
        if response:
            return response
        if default is not None:
            return str(default)
        print("Please enter a value.")


def prompt_float(prompt: str, default: float, low: float | None = None, high: float | None = None) -> float:
    while True:
        response = input(f"{prompt} [{default}]: ").strip()
        if not response:
            value = float(default)
        else:
            try:
                value = float(response)
            except ValueError:
                print("Please enter a number.")
                continue

        if low is not None and value < low:
            print(f"Please enter a value >= {low}.")
            continue
        if high is not None and value > high:
            print(f"Please enter a value <= {high}.")
            continue
        return value


def prompt_int(prompt: str, default: int, low: int | None = None, high: int | None = None) -> int:
    return int(prompt_float(prompt, default, low=low, high=high))


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        response = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def prompt_lineup(team_name: str, default_players: List[str]) -> List[str]:
    default_value = ", ".join(default_players)
    while True:
        raw = prompt_text(
            f"Enter the {team_name} lineup as comma-separated player names",
            default=default_value,
        )
        players = [player.strip() for player in raw.split(",") if player.strip()]
        deduped_players = list(dict.fromkeys(players))
        if deduped_players:
            return deduped_players
        print("Please enter at least one player.")


def collect_interactive_inputs(
    existing_ratings: pd.DataFrame,
    existing_lineups: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== INTERACTIVE LINEUP SETUP ===")
    print("First enter the lineup for each team. After that, you'll rate each player.")

    existing_lineups = existing_lineups.copy()
    existing_lineups["team"] = existing_lineups["team"].astype(str).str.upper()
    existing_lineups["position"] = existing_lineups["position"].astype(str).str.upper()

    sidemen_defaults = (
        existing_lineups[existing_lineups["team"] == "SIDEMEN"]["player"].astype(str).tolist()
        or [row["player"] for row in SEEDED_CURRENT_LINEUPS if row["team"] == "SIDEMEN"]
    )
    allstars_defaults = (
        existing_lineups[existing_lineups["team"] == "ALLSTARS"]["player"].astype(str).tolist()
        or [row["player"] for row in SEEDED_CURRENT_LINEUPS if row["team"] == "ALLSTARS"]
    )

    sidemen_players = prompt_lineup("SIDEMEN", sidemen_defaults)
    allstars_players = prompt_lineup("ALLSTARS", allstars_defaults)

    ratings_lookup = existing_ratings.set_index("player").to_dict(orient="index")
    lineup_lookup = {
        (str(row["team"]).upper(), str(row["player"])): row
        for _, row in existing_lineups.iterrows()
    }

    collected_lineups = []
    collected_ratings = []
    seen_players = set()

    for team_name, players in [("SIDEMEN", sidemen_players), ("ALLSTARS", allstars_players)]:
        print(f"\n=== {team_name} PLAYER DETAILS ===")
        for player in players:
            rating_defaults = ratings_lookup.get(player, {})
            lineup_defaults = lineup_lookup.get((team_name, player), {})

            print(f"\nPlayer: {player}")
            position = prompt_text(
                "Position (GK/DEF/MID/FWD)",
                default=str(lineup_defaults.get("position", rating_defaults.get("preferred_position", "MID"))).upper(),
            ).upper()
            expected_minutes = prompt_int(
                "Expected minutes",
                default=int(lineup_defaults.get("expected_minutes", rating_defaults.get("expected_minutes_default", 60))),
                low=0,
                high=90,
            )
            is_starter = 1 if prompt_yes_no(
                "Is this player a starter?",
                default=bool(int(lineup_defaults.get("is_starter", 1))),
            ) else 0

            collected_lineups.append({
                "match_label": TARGET_MATCH_LABEL,
                "team": team_name,
                "player": player,
                "expected_minutes": expected_minutes,
                "position": position,
                "is_starter": is_starter,
            })

            if player in seen_players:
                continue

            seen_players.add(player)
            print("Rate this player's attributes based on your own judgment.")

            collected_ratings.append({
                "player": player,
                "attack": prompt_int("Attack (0-100)", int(rating_defaults.get("attack", 50)), low=0, high=100),
                "defense": prompt_int("Defense (0-100)", int(rating_defaults.get("defense", 50)), low=0, high=100),
                "passing": prompt_int("Passing (0-100)", int(rating_defaults.get("passing", 50)), low=0, high=100),
                "fitness": prompt_int("Fitness (0-100)", int(rating_defaults.get("fitness", 50)), low=0, high=100),
                "goalkeeping": prompt_int("Goalkeeping (0-100)", int(rating_defaults.get("goalkeeping", 5)), low=0, high=100),
                "seriousness": prompt_int("Seriousness (0-100)", int(rating_defaults.get("seriousness", 50)), low=0, high=100),
                "volatility": prompt_int("Volatility (0-100)", int(rating_defaults.get("volatility", 50)), low=0, high=100),
                "experience": prompt_int("Experience (0-100)", int(rating_defaults.get("experience", 50)), low=0, high=100),
                "chemistry_core": prompt_int("Chemistry with team/core (0-100)", int(rating_defaults.get("chemistry_core", 50)), low=0, high=100),
                "expected_minutes_default": expected_minutes,
                "preferred_position": position,
            })

    ratings_df = pd.DataFrame(collected_ratings)
    lineups_df = pd.DataFrame(collected_lineups)
    return ratings_df, lineups_df


# -----------------------------
# Data validation
# -----------------------------

REQUIRED_PLAYER_COLUMNS = {
    "player", "attack", "defense", "passing", "fitness", "goalkeeping",
    "seriousness", "volatility", "experience", "chemistry_core",
    "expected_minutes_default", "preferred_position",
}

REQUIRED_LINEUP_COLUMNS = {
    "match_label", "team", "player", "expected_minutes", "position", "is_starter",
}

REQUIRED_HIST_COLUMNS = {
    "year", "sidemen_goals", "allstars_goals",
}


def validate_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


# -----------------------------
# Feature engineering
# -----------------------------

POSITION_ATTACK_WEIGHTS = {
    "GK": 0.05,
    "DEF": 0.40,
    "MID": 0.75,
    "FWD": 1.00,
}

POSITION_DEFENSE_WEIGHTS = {
    "GK": 1.10,
    "DEF": 1.00,
    "MID": 0.55,
    "FWD": 0.20,
}

POSITION_PASSING_WEIGHTS = {
    "GK": 0.20,
    "DEF": 0.60,
    "MID": 1.00,
    "FWD": 0.70,
}


@dataclass
class TeamFeatureSnapshot:
    team: str
    attack_total: float
    defense_total: float
    passing_total: float
    goalkeeper_quality: float
    fitness_avg: float
    seriousness_avg: float
    volatility_avg: float
    experience_avg: float
    chemistry_avg: float
    weighted_minutes_total: float
    bench_strength: float
    starter_strength: float
    shape_balance: float
    attack_defense_ratio: float
    top_heaviness: float
    chaos_factor: float
    lineup_size: int


def merge_lineups_with_ratings(
    lineups: pd.DataFrame,
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    merged = lineups.merge(player_ratings, on="player", how="left", validate="many_to_one")
    missing = merged[merged["attack"].isna()]["player"].unique().tolist()
    if missing:
        raise ValueError(
            "These lineup players are missing from player_ratings.csv: "
            + ", ".join(missing)
        )
    return merged


def compute_individual_row_strength(row: pd.Series) -> Dict[str, float]:
    pos = str(row["position"]).upper()
    minutes_weight = float(row["expected_minutes"]) / 90.0

    attack_component = row["attack"] * POSITION_ATTACK_WEIGHTS.get(pos, 0.7) * minutes_weight
    defense_component = row["defense"] * POSITION_DEFENSE_WEIGHTS.get(pos, 0.7) * minutes_weight
    passing_component = row["passing"] * POSITION_PASSING_WEIGHTS.get(pos, 0.7) * minutes_weight

    if pos == "GK":
        goalkeeper_component = row["goalkeeping"] * minutes_weight
    else:
        goalkeeper_component = 0.0

    general_value = (
        0.40 * row["attack"]
        + 0.25 * row["defense"]
        + 0.15 * row["passing"]
        + 0.10 * row["fitness"]
        + 0.05 * row["experience"]
        + 0.05 * row["seriousness"]
    ) * minutes_weight

    return {
        "attack_component": attack_component,
        "defense_component": defense_component,
        "passing_component": passing_component,
        "goalkeeper_component": goalkeeper_component,
        "general_value": general_value,
    }


def compute_team_features(team_df: pd.DataFrame, team_name: str) -> TeamFeatureSnapshot:
    if team_df.empty:
        raise ValueError(f"No players found for team {team_name}")

    augmented_rows = []
    for _, row in team_df.iterrows():
        metrics = compute_individual_row_strength(row)
        augmented_rows.append({**row.to_dict(), **metrics})

    tdf = pd.DataFrame(augmented_rows)

    starters = tdf[tdf["is_starter"] == 1].copy()
    bench = tdf[tdf["is_starter"] != 1].copy()

    attack_total = float(tdf["attack_component"].sum())
    defense_total = float(tdf["defense_component"].sum())
    passing_total = float(tdf["passing_component"].sum())
    goalkeeper_quality = float(tdf["goalkeeper_component"].sum())
    fitness_avg = float(np.average(tdf["fitness"], weights=np.maximum(tdf["expected_minutes"], 1)))
    seriousness_avg = float(np.average(tdf["seriousness"], weights=np.maximum(tdf["expected_minutes"], 1)))
    volatility_avg = float(np.average(tdf["volatility"], weights=np.maximum(tdf["expected_minutes"], 1)))
    experience_avg = float(np.average(tdf["experience"], weights=np.maximum(tdf["expected_minutes"], 1)))
    chemistry_avg = float(np.average(tdf["chemistry_core"], weights=np.maximum(tdf["expected_minutes"], 1)))
    weighted_minutes_total = float(tdf["expected_minutes"].sum())

    starter_strength = float(starters["general_value"].sum()) if not starters.empty else 0.0
    bench_strength = float(bench["general_value"].sum()) if not bench.empty else 0.0

    n_def = int((tdf["position"].str.upper() == "DEF").sum())
    n_mid = int((tdf["position"].str.upper() == "MID").sum())
    n_fwd = int((tdf["position"].str.upper() == "FWD").sum())
    n_gk = int((tdf["position"].str.upper() == "GK").sum())

    # Shape balance rewards presence of at least some defensive and midfield structure.
    shape_balance = (
        0.28 * min(n_gk, 1)
        + 0.28 * min(n_def / 2.0, 1.0)
        + 0.24 * min(n_mid / 2.0, 1.0)
        + 0.20 * min(n_fwd / 2.0, 1.0)
    ) * 100.0

    attack_defense_ratio = attack_total / max(defense_total, 1e-6)

    sorted_general = tdf["general_value"].sort_values(ascending=False).values
    top_two = sorted_general[:2].sum() if len(sorted_general) >= 2 else sorted_general.sum()
    overall = sorted_general.sum() if len(sorted_general) > 0 else 1.0
    top_heaviness = float(top_two / max(overall, 1e-6))

    # Chaos factor: more volatility + lower shape balance => more chaotic games
    chaos_factor = float(0.65 * volatility_avg + 0.35 * (100 - shape_balance))

    return TeamFeatureSnapshot(
        team=team_name,
        attack_total=attack_total,
        defense_total=defense_total,
        passing_total=passing_total,
        goalkeeper_quality=goalkeeper_quality,
        fitness_avg=fitness_avg,
        seriousness_avg=seriousness_avg,
        volatility_avg=volatility_avg,
        experience_avg=experience_avg,
        chemistry_avg=chemistry_avg,
        weighted_minutes_total=weighted_minutes_total,
        bench_strength=bench_strength,
        starter_strength=starter_strength,
        shape_balance=shape_balance,
        attack_defense_ratio=attack_defense_ratio,
        top_heaviness=top_heaviness,
        chaos_factor=chaos_factor,
        lineup_size=int(len(tdf)),
    )


def team_snapshot_to_dict(snapshot: TeamFeatureSnapshot) -> Dict[str, float]:
    return {
        "team": snapshot.team,
        "attack_total": snapshot.attack_total,
        "defense_total": snapshot.defense_total,
        "passing_total": snapshot.passing_total,
        "goalkeeper_quality": snapshot.goalkeeper_quality,
        "fitness_avg": snapshot.fitness_avg,
        "seriousness_avg": snapshot.seriousness_avg,
        "volatility_avg": snapshot.volatility_avg,
        "experience_avg": snapshot.experience_avg,
        "chemistry_avg": snapshot.chemistry_avg,
        "weighted_minutes_total": snapshot.weighted_minutes_total,
        "bench_strength": snapshot.bench_strength,
        "starter_strength": snapshot.starter_strength,
        "shape_balance": snapshot.shape_balance,
        "attack_defense_ratio": snapshot.attack_defense_ratio,
        "top_heaviness": snapshot.top_heaviness,
        "chaos_factor": snapshot.chaos_factor,
        "lineup_size": snapshot.lineup_size,
    }


# -----------------------------
# Historical expansion
# -----------------------------

def build_synthetic_training_frame(historical: pd.DataFrame) -> pd.DataFrame:
    """
    Because historical sample size is tiny, we create a small engineered training set
    around public past scores plus generic charity-match assumptions.

    This is NOT claiming precision; it is a pragmatic portfolio approach.
    """

    rows = []
    for _, row in historical.iterrows():
        s_goals = float(row["sidemen_goals"])
        a_goals = float(row["allstars_goals"])

        total_goals = s_goals + a_goals
        goal_diff = s_goals - a_goals

        # Approximate team latent strengths inferred from outcome.
        # These are pseudo-features to let a regression learn something.
        sid_attack = 72 + 2.4 * s_goals + 0.5 * goal_diff
        sid_defense = 66 + 0.8 * max(0, -a_goals + 6) + 0.4 * goal_diff
        sid_chem = 88
        sid_chaos = 62 + 0.9 * total_goals

        all_attack = 70 + 2.3 * a_goals - 0.5 * goal_diff
        all_defense = 63 + 0.7 * max(0, -s_goals + 6) - 0.4 * goal_diff
        all_chem = 76
        all_chaos = 64 + 0.9 * total_goals

        rows.append({
            "year": int(row["year"]),
            "team": "SIDEMEN",
            "team_attack": sid_attack,
            "team_defense": sid_defense,
            "team_chemistry": sid_chem,
            "team_chaos": sid_chaos,
            "opponent_attack": all_attack,
            "opponent_defense": all_defense,
            "opponent_chemistry": all_chem,
            "opponent_chaos": all_chaos,
            "goals": s_goals,
        })

        rows.append({
            "year": int(row["year"]),
            "team": "ALLSTARS",
            "team_attack": all_attack,
            "team_defense": all_defense,
            "team_chemistry": all_chem,
            "team_chaos": all_chaos,
            "opponent_attack": sid_attack,
            "opponent_defense": sid_defense,
            "opponent_chemistry": sid_chem,
            "opponent_chaos": sid_chaos,
            "goals": a_goals,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Modeling
# -----------------------------

def train_goal_regression_model(historical: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Trains a lightweight regression model on synthetic-expanded historical features.
    """
    train_df = build_synthetic_training_frame(historical)

    X = train_df[[
        "team_attack",
        "team_defense",
        "team_chemistry",
        "team_chaos",
        "opponent_attack",
        "opponent_defense",
        "opponent_chemistry",
        "opponent_chaos",
    ]].copy()

    y = train_df["goals"].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X, y)

    preds = model.predict(X)
    train_df["pred_goals"] = preds
    train_mae = mean_absolute_error(y, preds)
    print(f"[model] Synthetic-train MAE: {train_mae:.3f}")

    return model, train_df


def heuristic_expected_goals(
    own: TeamFeatureSnapshot,
    opp: TeamFeatureSnapshot,
) -> float:
    """
    Sparse-data heuristic that usually works surprisingly well for exhibition matches.
    Output is expected goals in roughly a 0.5 to 10.5 band.
    """

    base = 4.6  # these matches tend to be high-scoring

    attack_term = 0.040 * own.attack_total
    passing_term = 0.012 * own.passing_total
    fitness_term = 0.010 * (own.fitness_avg - 50)
    seriousness_term = 0.009 * (own.seriousness_avg - 50)
    chemistry_term = 0.010 * (own.chemistry_avg - 50)
    bench_term = 0.010 * own.bench_strength

    opp_def_term = -0.030 * opp.defense_total
    opp_gk_term = -0.018 * opp.goalkeeper_quality
    opp_shape_term = -0.010 * (opp.shape_balance - 50)

    chaos_boost = 0.010 * ((own.chaos_factor + opp.chaos_factor) / 2.0 - 50)
    top_heavy_adjustment = -0.45 * max(0.0, own.top_heaviness - 0.43)

    xg = (
        base
        + attack_term
        + passing_term
        + fitness_term
        + seriousness_term
        + chemistry_term
        + bench_term
        + opp_def_term
        + opp_gk_term
        + opp_shape_term
        + chaos_boost
        + top_heavy_adjustment
    )

    return clamp(xg, 0.5, 10.5)


def model_based_expected_goals(
    model: Pipeline,
    own: TeamFeatureSnapshot,
    opp: TeamFeatureSnapshot,
) -> float:
    pseudo_x = pd.DataFrame([{
        "team_attack": own.attack_total,
        "team_defense": own.defense_total,
        "team_chemistry": own.chemistry_avg,
        "team_chaos": own.chaos_factor,
        "opponent_attack": opp.attack_total,
        "opponent_defense": opp.defense_total,
        "opponent_chemistry": opp.chemistry_avg,
        "opponent_chaos": opp.chaos_factor,
    }])
    pred = float(model.predict(pseudo_x)[0])

    # Blend with heuristic because data is sparse
    heuristic = heuristic_expected_goals(own, opp)
    blended = 0.40 * pred + 0.60 * heuristic
    return clamp(blended, 0.5, 10.5)


# -----------------------------
# Simulation
# -----------------------------

def simulate_match(
    sidemen_xg: float,
    allstars_xg: float,
    sidemen_snapshot: TeamFeatureSnapshot,
    allstars_snapshot: TeamFeatureSnapshot,
    n_sims: int = SIMULATIONS,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simulate match outcomes using Poisson goals.
    Add slight overdispersion via gamma-Poisson mixture feel using volatility.
    """
    sid_chaos = sidemen_snapshot.chaos_factor
    all_chaos = allstars_snapshot.chaos_factor

    sid_sigma = 1.0 + (sid_chaos / 300.0)
    all_sigma = 1.0 + (all_chaos / 300.0)

    records = []
    sid_wins = 0
    all_wins = 0
    draws = 0

    for _ in range(n_sims):
        sid_lambda = np.random.lognormal(mean=np.log(max(sidemen_xg, 0.1)), sigma=0.08 * sid_sigma)
        all_lambda = np.random.lognormal(mean=np.log(max(allstars_xg, 0.1)), sigma=0.08 * all_sigma)

        sid_goals = np.random.poisson(sid_lambda)
        all_goals = np.random.poisson(all_lambda)

        if sid_goals > all_goals:
            sid_wins += 1
        elif sid_goals < all_goals:
            all_wins += 1
        else:
            draws += 1

        records.append({"sidemen_goals": sid_goals, "allstars_goals": all_goals})

    sim_df = pd.DataFrame(records)

    summary = {
        "sidemen_win_prob": sid_wins / n_sims,
        "allstars_win_prob": all_wins / n_sims,
        "draw_prob": draws / n_sims,
        "sidemen_avg_goals": float(sim_df["sidemen_goals"].mean()),
        "allstars_avg_goals": float(sim_df["allstars_goals"].mean()),
    }

    return sim_df, summary


def most_likely_scoreline(sim_df: pd.DataFrame) -> Tuple[int, int, int]:
    grouped = (
        sim_df.groupby(["sidemen_goals", "allstars_goals"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top = grouped.iloc[0]
    return int(top["sidemen_goals"]), int(top["allstars_goals"]), int(top["count"])


# -----------------------------
# Reporting
# -----------------------------

def save_probability_chart(summary: Dict[str, float]) -> None:
    labels = ["Sidemen", "Allstars", "Draw"]
    values = [
        summary["sidemen_win_prob"],
        summary["allstars_win_prob"],
        summary["draw_prob"],
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.title("Sidemen Charity Match Win Probabilities")
    plt.ylabel("Probability")
    plt.ylim(0, 1)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.1%}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "win_probabilities.png", dpi=200)
    plt.close()


def create_player_contribution_table(team_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in team_df.iterrows():
        metrics = compute_individual_row_strength(row)
        contribution = (
            0.45 * metrics["attack_component"]
            + 0.30 * metrics["defense_component"]
            + 0.15 * metrics["passing_component"]
            + 0.10 * metrics["goalkeeper_component"]
        )
        rows.append({
            "team": row["team"],
            "player": row["player"],
            "position": row["position"],
            "expected_minutes": row["expected_minutes"],
            "attack": row["attack"],
            "defense": row["defense"],
            "passing": row["passing"],
            "fitness": row["fitness"],
            "goalkeeping": row["goalkeeping"],
            "seriousness": row["seriousness"],
            "volatility": row["volatility"],
            "experience": row["experience"],
            "chemistry_core": row["chemistry_core"],
            "estimated_contribution": contribution,
        })
    return pd.DataFrame(rows).sort_values(["team", "estimated_contribution"], ascending=[True, False])


def write_summary_json(payload: Dict) -> None:
    with open(OUTPUT_DIR / "match_prediction_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    print("Setting up project...")
    ensure_directories()
    create_seed_files_if_missing()

    historical = pd.read_csv(HISTORICAL_MATCHES_CSV)
    player_ratings = pd.read_csv(PLAYER_RATINGS_CSV)
    current_lineups = pd.read_csv(CURRENT_LINEUPS_CSV)

    validate_columns(historical, REQUIRED_HIST_COLUMNS, "historical_matches.csv")
    validate_columns(player_ratings, REQUIRED_PLAYER_COLUMNS, "player_ratings.csv")
    validate_columns(current_lineups, REQUIRED_LINEUP_COLUMNS, "current_lineups.csv")

    if sys.stdin.isatty():
        use_interactive_setup = prompt_yes_no(
            "Would you like to enter the lineups and player ratings interactively now?",
            default=True,
        )
        if use_interactive_setup:
            player_ratings, current_lineups = collect_interactive_inputs(player_ratings, current_lineups)
            player_ratings.to_csv(PLAYER_RATINGS_CSV, index=False)
            current_lineups.to_csv(CURRENT_LINEUPS_CSV, index=False)
            print(f"Saved updated player ratings to {PLAYER_RATINGS_CSV}")
            print(f"Saved updated lineups to {CURRENT_LINEUPS_CSV}")
    else:
        print("Non-interactive run detected; using existing CSV inputs.")

    current_lineups["team"] = current_lineups["team"].str.upper()
    current_lineups["position"] = current_lineups["position"].str.upper()

    if not {"SIDEMEN", "ALLSTARS"}.issubset(set(current_lineups["team"].unique())):
        raise ValueError("current_lineups.csv must contain both SIDEMEN and ALLSTARS")

    merged = merge_lineups_with_ratings(current_lineups, player_ratings)

    sidemen_df = merged[merged["team"] == "SIDEMEN"].copy()
    allstars_df = merged[merged["team"] == "ALLSTARS"].copy()

    sid_snapshot = compute_team_features(sidemen_df, "SIDEMEN")
    all_snapshot = compute_team_features(allstars_df, "ALLSTARS")

    team_snapshot_df = pd.DataFrame([
        team_snapshot_to_dict(sid_snapshot),
        team_snapshot_to_dict(all_snapshot),
    ])
    team_snapshot_df.to_csv(OUTPUT_DIR / "team_feature_snapshot.csv", index=False)

    contrib_df = create_player_contribution_table(merged)
    contrib_df.to_csv(OUTPUT_DIR / "player_contributions.csv", index=False)

    model, synthetic_train_df = train_goal_regression_model(historical)
    synthetic_train_df.to_csv(OUTPUT_DIR / "synthetic_training_frame.csv", index=False)

    sid_xg_heur = heuristic_expected_goals(sid_snapshot, all_snapshot)
    all_xg_heur = heuristic_expected_goals(all_snapshot, sid_snapshot)

    sid_xg_model = model_based_expected_goals(model, sid_snapshot, all_snapshot)
    all_xg_model = model_based_expected_goals(model, all_snapshot, sid_snapshot)

    # Final blend
    sid_xg_final = clamp(0.50 * sid_xg_heur + 0.50 * sid_xg_model, 0.5, 10.5)
    all_xg_final = clamp(0.50 * all_xg_heur + 0.50 * all_xg_model, 0.5, 10.5)

    sim_df, sim_summary = simulate_match(
        sidemen_xg=sid_xg_final,
        allstars_xg=all_xg_final,
        sidemen_snapshot=sid_snapshot,
        allstars_snapshot=all_snapshot,
        n_sims=SIMULATIONS,
    )

    sim_df.to_csv(OUTPUT_DIR / "simulation_score_distribution.csv", index=False)

    likely_sid, likely_all, likely_count = most_likely_scoreline(sim_df)

    # Scorer-proxy probabilities using lineup contributions
    player_contrib = contrib_df.copy()
    player_contrib["team_total_contribution"] = player_contrib.groupby("team")["estimated_contribution"].transform("sum")
    player_contrib["share_of_team_contribution"] = (
        player_contrib["estimated_contribution"] / player_contrib["team_total_contribution"].replace(0, np.nan)
    ).fillna(0.0)

    # Rough scorer probability proxy: contribution share * expected goals, squashed to 0-1
    scorer_probs = []
    for _, row in player_contrib.iterrows():
        team_xg = sid_xg_final if row["team"] == "SIDEMEN" else all_xg_final
        raw = row["share_of_team_contribution"] * team_xg
        prob_score_at_least_one = 1 - math.exp(-0.85 * raw)
        scorer_probs.append({
            "team": row["team"],
            "player": row["player"],
            "score_at_least_one_prob": clamp(prob_score_at_least_one, 0.0, 0.95),
        })

    scorer_probs_df = pd.DataFrame(scorer_probs).sort_values(
        ["team", "score_at_least_one_prob"], ascending=[True, False]
    )
    scorer_probs_df.to_csv(OUTPUT_DIR / "player_scorer_probabilities.csv", index=False)

    save_probability_chart(sim_summary)

    payload = {
        "match_label": TARGET_MATCH_LABEL,
        "simulations": SIMULATIONS,
        "expected_goals": {
            "sidemen": round(sid_xg_final, 3),
            "allstars": round(all_xg_final, 3),
            "sidemen_heuristic": round(sid_xg_heur, 3),
            "allstars_heuristic": round(all_xg_heur, 3),
            "sidemen_model_blend_input": round(sid_xg_model, 3),
            "allstars_model_blend_input": round(all_xg_model, 3),
        },
        "win_probabilities": {
            "sidemen": round(sim_summary["sidemen_win_prob"], 4),
            "allstars": round(sim_summary["allstars_win_prob"], 4),
            "draw": round(sim_summary["draw_prob"], 4),
        },
        "average_goals": {
            "sidemen": round(sim_summary["sidemen_avg_goals"], 4),
            "allstars": round(sim_summary["allstars_avg_goals"], 4),
        },
        "most_likely_scoreline": {
            "sidemen_goals": likely_sid,
            "allstars_goals": likely_all,
            "sim_count": likely_count,
        },
        "team_feature_snapshot": [
            team_snapshot_to_dict(sid_snapshot),
            team_snapshot_to_dict(all_snapshot),
        ],
        "top_scorer_candidates": {
            "sidemen": scorer_probs_df[scorer_probs_df["team"] == "SIDEMEN"].head(5).to_dict(orient="records"),
            "allstars": scorer_probs_df[scorer_probs_df["team"] == "ALLSTARS"].head(5).to_dict(orient="records"),
        },
    }

    write_summary_json(payload)

    print("\n=== MATCH PREDICTION SUMMARY ===")
    print(f"Sidemen xG:  {sid_xg_final:.2f}")
    print(f"Allstars xG: {all_xg_final:.2f}")
    print(f"Sidemen win prob:  {sim_summary['sidemen_win_prob']:.1%}")
    print(f"Allstars win prob: {sim_summary['allstars_win_prob']:.1%}")
    print(f"Draw prob:         {sim_summary['draw_prob']:.1%}")
    print(f"Most likely score: {likely_sid}-{likely_all}")
    print("\nTop Sidemen scorers:")
    print(scorer_probs_df[scorer_probs_df["team"] == "SIDEMEN"].head(5).to_string(index=False))
    print("\nTop Allstars scorers:")
    print(scorer_probs_df[scorer_probs_df["team"] == "ALLSTARS"].head(5).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
