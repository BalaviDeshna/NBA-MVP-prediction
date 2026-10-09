"""Shared data loading + feature engineering for the NBA MVP project."""
import numpy as np
import pandas as pd

# Per-game, advanced and team-level features used by the models.
FEATURES = [
    "PTS", "TRB", "AST", "STL", "BLK", "TOV",
    "FG%", "3P%", "FT%", "MP", "G",
    "PER", "TS%", "USG%", "WS", "WS/48", "BPM", "VORP",
    "TeamWinPct",
]
TARGET = "Share"
MIN_GAMES = 41  # voters rarely consider players who missed half the season


def load_dataset(path: str = "data/dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only plausible MVP candidates and add season-relative features.

    Relative features matter because the league changes over time (pace, 3-point
    era): a 25 PPG season meant something different in 1990 vs 2020. Each stat is
    also expressed as a rank-percentile *within its season*.
    """
    df = df[df["G"] >= MIN_GAMES].copy()
    df[FEATURES] = df[FEATURES].fillna(0.0)

    # Candidate pool: top 15 players per season by Win Shares + anyone who got votes.
    df["ws_rank"] = df.groupby("Season")["WS"].rank(ascending=False, method="first")
    df = df[(df["ws_rank"] <= 15) | (df[TARGET] > 0)].copy()

    for col in ["PTS", "TRB", "AST", "PER", "WS", "BPM", "VORP"]:
        df[f"{col}_rel"] = df.groupby("Season")[col].rank(pct=True)
    return df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    return FEATURES + [c for c in df.columns if c.endswith("_rel")]


def top_k_accuracy(df: pd.DataFrame, pred_col: str, k: int = 1) -> float:
    """Fraction of seasons where the true MVP (max Share) is in the model's top-k."""
    hits, seasons = 0, 0
    for _, g in df.groupby("Season"):
        if g[TARGET].max() <= 0:
            continue
        true_mvp = g.loc[g[TARGET].idxmax(), "Player"]
        top = g.nlargest(k, pred_col)["Player"].tolist()
        hits += int(true_mvp in top)
        seasons += 1
    return hits / max(seasons, 1)
