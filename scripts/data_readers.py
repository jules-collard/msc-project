import glob
from itertools import chain

import orjson
import polars as pl
from polars import col as c


def batch_read_events(*patterns: str, lazy=True, **kwargs) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple event files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/NHL_..._sapifullevents.json"
    Returns a single polars dataframe with all event data.
    """

    df = (
        pl.scan_ndjson(
            *patterns,
            include_file_paths="source_path",
            **kwargs
        ).select(c('events', 'source_path'))
        .explode('events')
        .unnest('events')
        .with_columns(
            sportlogiq_game_id(c('source_path')),
            -c('y_coord', 'y_adj_coord'), # Event data y is flipped w.r.t. tracking data
            c('period').cast(pl.Int32),
        ).drop(c('source_path'))
    )
    return df if lazy else df.collect()


def batch_read_entity_tracking(patterns: list[str], mapping: pl.LazyFrame, lazy=True, **kwargs) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple entity tracking files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/NHL_..._entity_tracking_processed_measurements.parquet"
    Returns a single polars dataframe with all entity tracking data.
    """

    df = (
        pl.scan_parquet(
            patterns, 
            include_file_paths='source_path',
            **kwargs
        ).with_columns(
            smt_game_id(c('source_path')),
        ).drop(c('source_path'))
        .join(mapping.select(c('SportlogiqGameID', 'SMTGameID')), left_on='smt_game_id', right_on='SMTGameID', how='left')
        .drop(c('smt_game_id'))
        .rename({'SportlogiqGameID': 'game_id'})
    )
    return df if lazy else df.collect()

def batch_read_puck_tracking(patterns: list[str], mapping: pl.LazyFrame, lazy=True, extract_period_from_source=False, **kwargs) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple puck tracking files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/HOCKEY_NHL_..._Period_{period}.parquet"
    Returns a single polars dataframe with all puck tracking data.
    """

    df = (
        pl.scan_parquet(
            patterns, 
            include_file_paths="source_path",
            **kwargs
        ).with_columns(
            smt_game_id(c('source_path')),
        )
    )

    if extract_period_from_source: # Only for local development where files are split by period
        df = df.with_columns(extract_period("source_path"))

    df = (
        df
        .drop(c("source_path"))
        .join(mapping.select(c('SportlogiqGameID', 'SMTGameID')), left_on='smt_game_id', right_on='SMTGameID', how='left')
        .drop(c('smt_game_id'))
        .rename({'SportlogiqGameID': 'game_id'})
        .with_columns(
            c('game_id').cast(pl.String),
            c('z', 'vz', 'az').fill_null(0) # missing z values represent 0 (i.e. puck on ice)
        )
    )
    return df if lazy else df.collect()


def batch_read_rosters(patterns: list[str], lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple roster files at once, using a glob pattern.
    Returns a single polars dataframe with player information, with one row
    per player.
    """

    normalised_data = []

    # Create iterator of all files matching the patterns
    files = chain.from_iterable(glob.iglob(p) for p in patterns)

    for file_path in files:
        with open(file_path, "rb") as f:
            data = orjson.loads(f.read())

        for team_id, team_data in data.items():
            normalised_data.append({
                "team_id": team_id,
                "team_data": team_data
            })

    rosters = (
        pl.from_dicts(normalised_data)
        .explode(c('team_data'))
        .unnest()
        .drop(c('role', 'college'))
        .rename({'id': 'SportlogiqPlayerID'})
        .unique(c('SportlogiqPlayerID'))
    )

    return rosters.lazy() if lazy else rosters

def read_id_mapping(path: str) -> dict[str, str]:
    """
    Function to read NHL/Sportlogiq player ID mappings, returning a dictionary of Sportlogiq -> NHL ID mappings.
    """

    mapping_dict = ( 
        pl.read_csv(path)
        .drop(c('PlayerName'))
        .cast(pl.String)
        .rows_by_key('SportlogiqPlayerID', unique=True)
    )

    return {key: value[0] for key, value in mapping_dict.items()}

def read_game_id_mapping(path: str) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .with_columns(c('GameDate').str.strptime(pl.Date, format="%Y-%m-%d"))
    )

def sportlogiq_game_id(source_expr: pl.Expr | str) -> pl.Expr:
    """
    Function to extract game_id from a column containing file paths.
    """
    source_expr = c(source_expr) if isinstance(source_expr, str) else source_expr
    return source_expr.str.extract(r"/games/(\d+)/").cast(pl.String).alias("game_id")

def smt_game_id(source_expr: pl.Expr | str) -> pl.Expr:
    """
    Function to extract game_id from a column containing file paths.
    """
    source_expr = c(source_expr) if isinstance(source_expr, str) else source_expr
    return source_expr.str.extract(r"/games/([^/]+)/").cast(pl.String).alias("smt_game_id")

def extract_period(col_name: str) -> pl.Expr:
    """
    Function to extract period from a column containing file paths.
    """
    return c(col_name).str.extract(r"period_(\d+)").cast(pl.Int32).alias("period")