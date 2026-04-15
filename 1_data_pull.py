"""
STEP 1 — DATA PULL
==================
Pulls play-by-play data for a full NBA season using nba_api (PlayByPlayV3).
Saves raw play-by-play CSVs to data/raw/.

Usage:
    python 1_data_pull.py --season 2022-23 --max_games 100
    python 1_data_pull.py --season 2022-23 --max_games 0   # full season
    python 1_data_pull.py --season 2022-23 --team GSW      # only Warriors games
    python 1_data_pull.py --season 2022-23 --team "Golden State Warriors"
"""

import os
import time
import argparse
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
DELAY = 0.7


def get_game_ids(season, team=None):
    print(f"Fetching game list for {season}...")
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    games_df = finder.get_data_frames()[0]

    if team:
        team_upper = team.upper()
        mask = (
            games_df["TEAM_ABBREVIATION"].str.upper() == team_upper
        ) | (
            games_df["TEAM_NAME"].str.upper() == team_upper
        )
        games_df = games_df[mask]
        if games_df.empty:
            raise ValueError(
                f"No games found for team '{team}'. "
                "Check the abbreviation (e.g. GSW, LAL, BOS) or full team name."
            )
        print(f"  Filtering to team '{team}'.")

    game_ids = games_df["GAME_ID"].unique().tolist()
    print(f"  Found {len(game_ids)} games.")
    return game_ids


def fetch_pbp(game_id):
    """
    Fetch play-by-play using PlayByPlayV3.
    V3 columns we care about:
        actionType  : "Made Shot" | "Missed Shot" | "Turnover" | "Free Throw" | ...
        subType     : "Driving Layup Shot" | "Step Back Jump Shot" | etc.
        description : full text e.g. "Curry 26' 3PT Pull-Up Jump Shot (3 PTS)"
        shotResult  : "Made" | "Missed"
    """
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
        df  = pbp.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"  WARNING {game_id}: {e}")
        return None


def main(season, max_games, team=None):
    game_ids = get_game_ids(season, team=team)
    if max_games:
        game_ids = game_ids[:max_games]
        print(f"  Limiting to {max_games} games.")

    print(f"\nDownloading play-by-play for {len(game_ids)} games...\n")
    success, skipped, failed = 0, 0, 0

    for game_id in tqdm(game_ids):
        out_path = os.path.join(RAW_DIR, f"{game_id}.csv")
        if os.path.exists(out_path):
            skipped += 1
            continue
        df = fetch_pbp(game_id)
        if df is not None and not df.empty:
            df.to_csv(out_path, index=False)
            success += 1
        else:
            failed += 1
        time.sleep(DELAY)

    print(f"\nDone.  Success: {success}  Skipped: {skipped}  Failed: {failed}")
    print(f"Raw files saved to: {RAW_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",    default="2022-23") 
    parser.add_argument("--max_games", default=100, type=int,
                        help="0 = full season, otherwise limit e.g. 100")
    parser.add_argument("--team", default=None,
                        help="Filter to a specific team, e.g. GSW or 'Golden State Warriors'")
    args = parser.parse_args()
    main(args.season, args.max_games, team=args.team)