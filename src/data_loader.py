"""
Data loading module for European Soccer Database.

Functions to connect to SQLite database, extract match data, team information,
and filter for top 5 European leagues.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .config import DATABASE_PATH, TOP_5_LEAGUES


def connect_to_database(db_path: str = DATABASE_PATH) -> sqlite3.Connection:
    """
    Establish connection to SQLite database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object

    Raises:
        FileNotFoundError: If database file doesn't exist
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(db_path)


def get_table_names(conn: sqlite3.Connection) -> List[str]:
    """
    Get list of all table names in the database.

    Args:
        conn: SQLite connection object

    Returns:
        List of table names
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def load_matches(conn: sqlite3.Connection,
                 league_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load match data from database, optionally filtered by league.

    Args:
        conn: SQLite connection object
        league_ids: List of league IDs to filter by. If None, loads all matches.

    Returns:
        DataFrame containing match data
    """
    if league_ids is None:
        query = "SELECT * FROM Match"
    else:
        placeholders = ",".join("?" * len(league_ids))
        query = f"SELECT * FROM Match WHERE league_id IN ({placeholders})"
        return pd.read_sql_query(query, conn, params=league_ids)

    return pd.read_sql_query(query, conn)


def load_teams(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load team information from database.

    Args:
        conn: SQLite connection object

    Returns:
        DataFrame containing team data
    """
    return pd.read_sql_query("SELECT * FROM Team", conn)


def load_team_attributes(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load team attributes (FIFA-based tactical attributes).

    Args:
        conn: SQLite connection object

    Returns:
        DataFrame containing team attributes
    """
    return pd.read_sql_query("SELECT * FROM Team_Attributes", conn)


def load_league_info(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load league and country information.

    Args:
        conn: SQLite connection object

    Returns:
        DataFrame containing league information
    """
    query = """
    SELECT l.id as league_id, l.name as league_name,
           c.id as country_id, c.name as country_name
    FROM League l
    JOIN Country c ON l.country_id = c.id
    """
    return pd.read_sql_query(query, conn)


def load_top5_matches(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load matches from top 5 European leagues only.

    Args:
        conn: SQLite connection object

    Returns:
        DataFrame containing match data for top 5 leagues
    """
    league_ids = list(TOP_5_LEAGUES.values())
    return load_matches(conn, league_ids)
