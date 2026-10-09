"""Collect NBA season data from Basketball-Reference and build data/dataset.csv.

For every season it downloads:
  * per-game stats       (PTS, TRB, AST, STL, BLK, FG%, ...)
  * advanced stats       (PER, TS%, WS, BPM, VORP, USG%, ...)
  * team standings       (team win percentage)
  * MVP award voting     (the target: vote share)

Be polite to the site: requests are rate limited (SLEEP seconds apart) and raw
HTML is cached in data/raw/ so each page is downloaded only once.

Usage:
    python src/collect_data.py --start 1980 --end 2025
"""
import argparse
import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.basketball-reference.com"
RAW = Path("data/raw")
SLEEP = 4  # seconds between uncached requests (site allows ~20 req/min)
HEADERS = {"User-Agent": "Mozilla/5.0 (university ML course project)"}


def fetch(url: str, cache_name: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / cache_name
    if path.exists():
        return path.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # Some tables are wrapped in HTML comments; un-comment so read_html sees them.
    html = resp.text.replace("<!--", "").replace("-->", "")
    path.write_text(html, encoding="utf-8")
    time.sleep(SLEEP)
    return html


def read_table(html: str, table_id: str) -> pd.DataFrame:
    return pd.read_html(StringIO(html), attrs={"id": table_id})[0]


def clean_player_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Player"] != "Player"].copy()  # repeated header rows
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    df["Player"] = df["Player"].str.replace("*", "", regex=False).str.strip()
    return df


def dedupe_traded(df: pd.DataFrame) -> pd.DataFrame:
    """Players traded mid-season have a 'TOT' row plus one row per team.
    Keep the TOT row for stats, and record the *last* team for team-strength lookup."""
    last_team = df[df["Tm"] != "TOT"].groupby("Player")["Tm"].last()
    out = df[(df["Tm"] == "TOT") | (~df["Player"].duplicated(keep=False))].copy()
    out["Team"] = out["Player"].map(last_team).fillna(out["Tm"])
    return out


def get_team_win_pct(year: int) -> pd.DataFrame:
    html = fetch(f"{BASE}/leagues/NBA_{year}_standings.html", f"standings_{year}.html")
    rows = []
    for tid in ("divs_standings_E", "divs_standings_W", "divs_standings_"):
        try:
            tbl = pd.read_html(StringIO(html), attrs={"id": tid}, extract_links="body")[0]
        except Exception:
            continue
        name_col = tbl.columns[0]
        for _, r in tbl.iterrows():
            text, href = r[name_col]
            if href and "/teams/" in href:
                abbr = href.split("/")[2]
                wl = r.get("W/L%")
                rows.append({"Team": abbr, "TeamWinPct": float(wl[0]) if isinstance(wl, tuple) else float(wl)})
    # Older seasons use a single combined table
    return pd.DataFrame(rows).drop_duplicates("Team")


def get_mvp_votes(year: int) -> pd.DataFrame:
    html = fetch(f"{BASE}/awards/awards_{year}.html", f"awards_{year}.html")
    df = read_table(html, "mvp")
    df.columns = [c[1] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Player", "Share"]].copy()
    df["Player"] = df["Player"].str.replace("*", "", regex=False).str.strip()
    return df


def build_season(year: int) -> pd.DataFrame:
    per_game = clean_player_table(read_table(fetch(f"{BASE}/leagues/NBA_{year}_per_game.html", f"pergame_{year}.html"), "per_game_stats"))
    adv = clean_player_table(read_table(fetch(f"{BASE}/leagues/NBA_{year}_advanced.html", f"advanced_{year}.html"), "advanced_stats"))

    per_game = dedupe_traded(per_game)
    adv = adv[(adv["Tm"] == "TOT") | (~adv["Player"].duplicated(keep=False))]
    adv = adv.drop(columns=["Rk", "Pos", "Age", "Tm", "G", "MP"], errors="ignore")
    df = per_game.merge(adv, on="Player", how="left", suffixes=("", "_adv"))

    df = df.merge(get_team_win_pct(year), on="Team", how="left")
    df = df.merge(get_mvp_votes(year), on="Player", how="left")
    df["Share"] = df["Share"].fillna(0.0)
    df["Season"] = year
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1980)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--out", default="data/dataset.csv")
    args = ap.parse_args()

    frames = []
    for year in range(args.start, args.end + 1):
        print(f"Season {year-1}-{str(year)[2:]} ...", flush=True)
        try:
            frames.append(build_season(year))
        except Exception as e:  # keep going; report at end
            print(f"  FAILED: {e}")
    data = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"Saved {len(data)} rows, {data['Season'].nunique()} seasons -> {args.out}")


if __name__ == "__main__":
    main()
