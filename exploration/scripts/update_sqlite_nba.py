#!/usr/bin/env python3
"""
update_sqlite_nba.py

Pipeline (idempotent):
1) Pull LeagueGameLog for Season + SeasonType, PlayerOrTeam=T
2) Identify missing GAME_IDs vs SQLite (either table)
3) For each missing GAME_ID:
   - Pull boxscoretraditionalv3 (player rows)
   - Pull boxscoreadvancedv3 (nested players -> flatten statistics)
4) UPSERT into SQLite tables:
   - player_game_traditional
   - player_game_advanced

Notes:
- Uses nba_api wrappers (still hits stats.nba.com).
- Uses robust headers to reduce blocking.
- Uses UPSERT so you can safely rerun.
"""

from __future__ import annotations

import argparse
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamelog, boxscoretraditionalv3, boxscoreadvancedv3


# ----------------------------
# Config / Constants
# ----------------------------
DEFAULT_SEASON = "2025-26"
DEFAULT_SEASON_TYPE = "Regular Season"

DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 90

DB_PATH_DEFAULT = "data/nba.sqlite"

# Browser-like headers (helps with NBA throttling/blocks)
DEFAULT_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://stats.nba.com/",
    "Origin": "https://stats.nba.com",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Chromium";v="140", "Google Chrome";v="140", "Not;A=Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Site": "same-origin",
}


# ----------------------------
# Logging / utilities
# ----------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sleep_with_jitter(base_seconds: float) -> None:
    jitter = random.uniform(0.25, 0.85)
    time.sleep(max(0.0, base_seconds) + jitter)


def call_with_retries(fn, *, retries: int, sleep_s: float, what: str):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            logging.warning("%s failed (attempt %d/%d): %s", what, attempt, retries, e)
            sleep_with_jitter(sleep_s * (2 ** (attempt - 1)))
    raise RuntimeError(f"{what} failed after {retries} retries") from last_err


def safe_str(x) -> str:
    return str(x).strip()


# ----------------------------
# SQLite schema + helpers
# ----------------------------
TRAD_TABLE = "player_game_traditional"
ADV_TABLE = "player_game_advanced"


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TRAD_TABLE} (
        game_id TEXT NOT NULL,
        season INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        team_id INTEGER,
        team_tricode TEXT,
        team_side TEXT,
        first_name TEXT,
        family_name TEXT,
        name_i TEXT,
        player_slug TEXT,
        position TEXT,
        comment TEXT,
        jersey_num TEXT,
        minutes TEXT,

        field_goals_made INTEGER,
        field_goals_attempted INTEGER,
        field_goals_percentage REAL,
        three_pointers_made INTEGER,
        three_pointers_attempted INTEGER,
        three_pointers_percentage REAL,
        free_throws_made INTEGER,
        free_throws_attempted INTEGER,
        free_throws_percentage REAL,
        rebounds_offensive INTEGER,
        rebounds_defensive INTEGER,
        rebounds_total INTEGER,
        assists INTEGER,
        steals INTEGER,
        blocks INTEGER,
        turnovers INTEGER,
        fouls_personal INTEGER,
        points INTEGER,
        plus_minus_points INTEGER,

        PRIMARY KEY (game_id, person_id)
    );
    """)

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {ADV_TABLE} (
        game_id TEXT NOT NULL,
        season INTEGER NOT NULL,
        person_id INTEGER NOT NULL,
        team_id INTEGER,
        team_tricode TEXT,
        team_side TEXT,
        first_name TEXT,
        family_name TEXT,
        name_i TEXT,
        player_slug TEXT,
        position TEXT,
        comment TEXT,
        jersey_num TEXT,
        minutes TEXT,

        estimated_offensive_rating REAL,
        offensive_rating REAL,
        estimated_defensive_rating REAL,
        defensive_rating REAL,
        estimated_net_rating REAL,
        net_rating REAL,
        assist_percentage REAL,
        assist_to_turnover REAL,
        assist_ratio REAL,
        offensive_rebound_percentage REAL,
        defensive_rebound_percentage REAL,
        rebound_percentage REAL,
        turnover_ratio REAL,
        effective_field_goal_percentage REAL,
        true_shooting_percentage REAL,
        usage_percentage REAL,
        estimated_usage_percentage REAL,
        estimated_pace REAL,
        pace REAL,
        pace_per40 REAL,
        possessions REAL,
        pie REAL,

        PRIMARY KEY (game_id, person_id)
    );
    """)

    # Helpful indexes
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TRAD_TABLE}_person ON {TRAD_TABLE}(person_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TRAD_TABLE}_game   ON {TRAD_TABLE}(game_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{ADV_TABLE}_person  ON {ADV_TABLE}(person_id);")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{ADV_TABLE}_game    ON {ADV_TABLE}(game_id);")

    conn.commit()


def existing_game_ids(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT game_id FROM {TRAD_TABLE}")
    trad = {row[0] for row in cur.fetchall()}
    cur.execute(f"SELECT DISTINCT game_id FROM {ADV_TABLE}")
    adv = {row[0] for row in cur.fetchall()}
    return trad & adv


def upsert_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame, pk_cols: list[str]) -> None:
    if df.empty:
        return

    df = df.copy()

    # Force numpy-backed numeric dtypes to native Python types
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int64").astype(int)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float)
    # Ensure pk present
    for c in pk_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required PK col '{c}' for table {table}")

    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)

    # UPSERT: on PK conflict, update all non-PK columns
    non_pk = [c for c in cols if c not in pk_cols]
    update_set = ", ".join([f"{c}=excluded.{c}" for c in non_pk]) if non_pk else ""

    sql = f"""
    INSERT INTO {table} ({col_list})
    VALUES ({placeholders})
    ON CONFLICT({", ".join(pk_cols)}) DO UPDATE SET
    {update_set};
    """

    # Build rows ensuring numpy scalars are converted to native Python types
    rows = [
        tuple(
            v.item() if isinstance(v, np.generic) else v
            for v in df.loc[i, cols]
        )
        for i in df.index
    ]
    conn.executemany(sql, rows)
    conn.commit()


# ----------------------------
# Fetch functions
# ----------------------------
def fetch_league_game_log(
    season: str,
    season_type: str,
    *,
    sleep_s: float,
    retries: int,
    timeout_s: int,
    headers: Optional[dict] = None,
) -> pd.DataFrame:
    def _do():
        lgl = leaguegamelog.LeagueGameLog(
            counter=0,
            direction="ASC",
            league_id="00",
            player_or_team_abbreviation="T",
            season=season,
            season_type_all_star=season_type,
            sorter="DATE",
            timeout=timeout_s,
            headers=headers or DEFAULT_HEADERS,
        )
        return lgl.get_data_frames()[0]

    df = call_with_retries(_do, retries=retries, sleep_s=sleep_s, what="LeagueGameLog")
    sleep_with_jitter(sleep_s)
    return df


def fetch_boxscore_traditional(game_id: str, *, sleep_s: float, retries: int, timeout_s: int, headers: Optional[dict] = None) -> pd.DataFrame:
    gid = safe_str(game_id)

    def _do():
        bs = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=gid,
            start_period=0,
            end_period=0,
            start_range=0,
            end_range=0,
            range_type=0,
            timeout=timeout_s,
            headers=headers or DEFAULT_HEADERS,
        )
        dfs = bs.get_data_frames()
        player_df = dfs[0].copy()

        # Normalize/rename into our DB schema
        rename = {
            "gameId": "game_id",
            "teamId": "team_id",
            "teamTricode": "team_tricode",
            "teamSide": "team_side",
            "personId": "person_id",
            "firstName": "first_name",
            "familyName": "family_name",
            "nameI": "name_i",
            "playerSlug": "player_slug",
            "jerseyNum": "jersey_num",
            "fieldGoalsMade": "field_goals_made",
            "fieldGoalsAttempted": "field_goals_attempted",
            "fieldGoalsPercentage": "field_goals_percentage",
            "threePointersMade": "three_pointers_made",
            "threePointersAttempted": "three_pointers_attempted",
            "threePointersPercentage": "three_pointers_percentage",
            "freeThrowsMade": "free_throws_made",
            "freeThrowsAttempted": "free_throws_attempted",
            "freeThrowsPercentage": "free_throws_percentage",
            "reboundsOffensive": "rebounds_offensive",
            "reboundsDefensive": "rebounds_defensive",
            "reboundsTotal": "rebounds_total",
            "foulsPersonal": "fouls_personal",
            "plusMinusPoints": "plus_minus_points",
        }
        player_df = player_df.rename(columns={k: v for k, v in rename.items() if k in player_df.columns})

        # Ensure required columns exist
        if "game_id" not in player_df.columns:
            player_df["game_id"] = gid
        else:
            player_df["game_id"] = player_df["game_id"].astype(str)

        # Keep only columns that exist in our table schema
        keep = [
            "game_id","person_id","team_id","team_tricode","team_side",
            "first_name","family_name","name_i","player_slug","position","comment","jersey_num","minutes",
            "field_goals_made","field_goals_attempted","field_goals_percentage",
            "three_pointers_made","three_pointers_attempted","three_pointers_percentage",
            "free_throws_made","free_throws_attempted","free_throws_percentage",
            "rebounds_offensive","rebounds_defensive","rebounds_total",
            "assists","steals","blocks","turnovers","fouls_personal","points","plus_minus_points"
        ]
        player_df = player_df[[c for c in keep if c in player_df.columns]].copy()

        return player_df

    df = call_with_retries(_do, retries=retries, sleep_s=sleep_s, what=f"BoxScoreTraditionalV3({gid})")
    sleep_with_jitter(sleep_s)
    return df


def fetch_boxscore_advanced(game_id: str, *, sleep_s: float, retries: int, timeout_s: int, headers: Optional[dict] = None) -> pd.DataFrame:
    gid = safe_str(game_id)

    def _do():
        adv = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=gid,
            start_period=0,
            end_period=0,
            start_range=0,
            end_range=0,
            range_type=0,
            timeout=timeout_s,
            headers=headers or DEFAULT_HEADERS,
        )
        # nba_api v3 endpoints expose the raw dict via get_dict()
        payload = adv.get_dict()
        game = payload["boxScoreAdvanced"]
        game_id_local = game["gameId"]

        rows = []
        for team_side in ["homeTeam", "awayTeam"]:
            team = game[team_side]
            team_id = team.get("teamId")
            team_tricode = team.get("teamTricode")
            for p in team.get("players", []):
                stats = p.get("statistics", {}) or {}
                base = {k: v for k, v in p.items() if k != "statistics"}

                flat = {
                    "game_id": game_id_local,
                    "team_id": team_id,
                    "team_tricode": team_tricode,
                    "team_side": team_side.replace("Team", ""),
                    "person_id": base.get("personId"),
                    "first_name": base.get("firstName"),
                    "family_name": base.get("familyName"),
                    "name_i": base.get("nameI"),
                    "player_slug": base.get("playerSlug"),
                    "position": base.get("position"),
                    "comment": base.get("comment"),
                    "jersey_num": base.get("jerseyNum"),
                    "minutes": stats.get("minutes"),
                    "estimated_offensive_rating": stats.get("estimatedOffensiveRating"),
                    "offensive_rating": stats.get("offensiveRating"),
                    "estimated_defensive_rating": stats.get("estimatedDefensiveRating"),
                    "defensive_rating": stats.get("defensiveRating"),
                    "estimated_net_rating": stats.get("estimatedNetRating"),
                    "net_rating": stats.get("netRating"),
                    "assist_percentage": stats.get("assistPercentage"),
                    "assist_to_turnover": stats.get("assistToTurnover"),
                    "assist_ratio": stats.get("assistRatio"),
                    "offensive_rebound_percentage": stats.get("offensiveReboundPercentage"),
                    "defensive_rebound_percentage": stats.get("defensiveReboundPercentage"),
                    "rebound_percentage": stats.get("reboundPercentage"),
                    "turnover_ratio": stats.get("turnoverRatio"),
                    "effective_field_goal_percentage": stats.get("effectiveFieldGoalPercentage"),
                    "true_shooting_percentage": stats.get("trueShootingPercentage"),
                    "usage_percentage": stats.get("usagePercentage"),
                    "estimated_usage_percentage": stats.get("estimatedUsagePercentage"),
                    "estimated_pace": stats.get("estimatedPace"),
                    "pace": stats.get("pace"),
                    "pace_per40": stats.get("pacePer40"),
                    "possessions": stats.get("possessions"),
                    "pie": stats.get("PIE"),
                }
                rows.append(flat)

        df = pd.DataFrame(rows)
        return df

    df = call_with_retries(_do, retries=retries, sleep_s=sleep_s, what=f"BoxScoreAdvancedV3({gid})")
    sleep_with_jitter(sleep_s)
    return df


# ----------------------------
# Orchestration
# ----------------------------
def compute_missing_game_ids(lgl: pd.DataFrame, already: set[str]) -> list[str]:
    if lgl.empty or "GAME_ID" not in lgl.columns:
        return []
    game_ids = sorted({safe_str(x) for x in lgl["GAME_ID"].tolist()})
    missing = [gid for gid in game_ids if gid not in already]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Update SQLite DB with NBA boxscore traditional + advanced (v3).")
    parser.add_argument("--db", type=str, default=DB_PATH_DEFAULT, help="SQLite DB path (e.g. data/nba.sqlite)")
    parser.add_argument("--season", type=str, default=DEFAULT_SEASON, help='NBA season string, e.g. "2025-26"')
    parser.add_argument("--season-type", type=str, default=DEFAULT_SEASON_TYPE, help='Season type, e.g. "Regular Season"')
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep between API calls (seconds).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP read timeout per request (seconds).")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per API call.")
    parser.add_argument("--max-games", type=int, default=None, help="Optional limit for missing games backfill (debug).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)

    conn = connect(args.db)
    init_db(conn)

    already = existing_game_ids(conn)
    logging.info("DB has %d existing distinct game_ids.", len(already))

    # 1) Discover games from LeagueGameLog (Team mode)
    lgl = fetch_league_game_log(
        season=args.season,
        season_type=args.season_type,
        sleep_s=args.sleep,
        retries=args.retries,
        timeout_s=args.timeout,
        headers=DEFAULT_HEADERS,
    )

    missing = compute_missing_game_ids(lgl, already)

    if not missing:
        logging.info("No missing games. DB appears up to date for %s (%s).", args.season, args.season_type)
        conn.close()
        return 0

    if args.max_games is not None:
        missing = missing[: args.max_games]
        logging.info("Limiting to max_games=%d -> processing %d games.", args.max_games, len(missing))

    logging.info("Found %d missing GAME_IDs to backfill.", len(missing))

    # 2) Backfill each missing game
    for i, gid in enumerate(missing, start=1):
        logging.info("(%d/%d) Fetching GAME_ID=%s", i, len(missing), gid)
        # Periodic cooldown to reduce risk of rate limiting
        if i % 50 == 0:
            logging.info("Cooldown triggered after %d games. Sleeping 60 seconds...", i)
            time.sleep(60)

        try:
            df_t = fetch_boxscore_traditional(
                gid,
                sleep_s=args.sleep,
                retries=args.retries,
                timeout_s=args.timeout,
                headers=DEFAULT_HEADERS,
            )
            season_end = int(args.season[:2] + args.season[5:])
            df_t["season"] = season_end
            upsert_df(conn, TRAD_TABLE, df_t, pk_cols=["game_id", "person_id"])
            logging.info("Upserted traditional rows: %d", len(df_t))
        except Exception as e:
            logging.error("Traditional failed for %s: %s", gid, e)

        try:
            df_a = fetch_boxscore_advanced(
                gid,
                sleep_s=args.sleep,
                retries=args.retries,
                timeout_s=args.timeout,
                headers=DEFAULT_HEADERS,
            )
            season_end = int(args.season[:2] + args.season[5:])
            df_a["season"] = season_end
            upsert_df(conn, ADV_TABLE, df_a, pk_cols=["game_id", "person_id"])
            logging.info("Upserted advanced rows: %d", len(df_a))
        except Exception as e:
            logging.error("Advanced failed for %s: %s", gid, e)

    conn.close()
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())