from typing import List, Dict
import json

import polars as pl
from polars import col as c
import ijson

from parquet_helpers import EntityTrackingReader

def read_events(path: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read event data from *_sapifullevents.json type files.
    """

    with open(path, "r") as f:
        events_dict = json.load(f)
    
    df = (
        pl.from_dicts(events_dict.get("events"), infer_schema_length=None)
        .with_columns(-c('y_coord', 'y_adj_coord')) # Event data y is flipped w.r.t. tracking data
    )
    return df.lazy() if lazy else df

def batch_read_events(pattern: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple event files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/NHL_..._sapifullevents.json"
    Returns a single polars dataframe with all event data.
    """

    df = (
        pl.scan_ndjson(
            pattern,
            include_file_paths="source_path"
        ).select(c('events', 'source_path'))
        .explode('events')
        .unnest('events')
        .with_columns(
            extract_game_id('source_path', '.*_sapifullevents.json'),
        ).drop(c('source_path'))
    )
    return df if lazy else df.collect()

def read_entity_registration(path: str, lazy=True, clean=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read entity_registration.json files.
    Extracts roster information, with 1 row per player/entity.
    """

    with open(path) as f:
        rosters_json = json.load(f)

    rosters = pl.from_dicts(rosters_json[0].get("Entities"))

    if clean:
        rosters = (
            rosters
            .filter(c('EntityType') == 'Player')
            .select(c('EntityId', 'VisOrHome', 'JerseyNum', 'FirstName', 'LastName', 'Position', 'NAbbrev',
                      'EntityTeamId', 'OfficialId', 'Handed'))
        )

    return rosters.lazy() if lazy else rosters

def read_entity_tracking(path: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read tracking data from entity_tracking_processed_measurements.parquet files, returning
    a polars dataframe.
    """

    with open(path, mode="rb") as f:
        reader = EntityTrackingReader(f.read())

    df = pl.from_arrow(reader.get_table())
    return df.lazy() if lazy else df

def batch_read_entity_tracking(pattern: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple entity tracking files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/NHL_..._entity_tracking_processed_measurements.parquet"
    Returns a single polars dataframe with all entity tracking data.
    """

    df = (
        pl.scan_parquet(
            pattern, 
            include_file_paths='source_path'
        ).with_columns(
            extract_game_id('source_path', '.*_entity_tracking_processed_measurements\.parquet'),
        ).drop(c('source_path'))
    )
    return df if lazy else df.collect()

def read_puck_tracking(paths: List[str], periods: List[int], lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to parse raw puck tracking data. Takes list of filenames and corresponding
    periods, and returns 1 polars dataframe with puck tracking. Uses ijson and generators
    to avoid reading all files into memory at once.
    """
    
    def all_puck_objects():
        for path, period in zip(paths, periods):    
            with open(path, "r") as f:
                for o in ijson.items(f, 'item.TrackingData.item', use_float=True):
                    if o['EntityId'] == '1':
                        yield o, period

    def puck_with_period():
        for o, period in all_puck_objects():
            yield {**o, 'period': period}

    puck_tracking = (
        pl.from_dicts(puck_with_period()).lazy()
        .with_columns(
            c('Location').struct.rename_fields(['x', 'y', 'z']).struct.unnest(), 
            c('Velocity').struct.rename_fields(['vx', 'vy', 'vz']).struct.unnest(),
            c('Acceleration').struct.rename_fields(['ax', 'ay', 'az']).struct.unnest(),
        ).with_columns(
            c('z', 'vz', 'az').fill_null(0) # missing z values represent 0 (i.e. puck on ice)
        ).drop(c('Location', 'Velocity', 'Acceleration', 'OnPlayingSurface', 'PayloadData', 'EntityOfficialId',
                'Landmarks3D', 'MetaTag1', 'LocationLTC', 'MeasurementId', 'LocationConfidence'))
        .rename({'LocationUTC':'ts', 'ClockState':'clock_state', 'EntityId':'entity_id'})
    )

    return puck_tracking if lazy else puck_tracking.collect()

def batch_read_puck_tracking(pattern: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read multiple puck tracking files at once, using a glob pattern.
    Expects file names in the format "data/{game_id}/HOCKEY_NHL_..._Period_{period}.parquet"
    Returns a single polars dataframe with all puck tracking data.
    """

    df = (
        pl.scan_parquet(
            pattern, 
            include_file_paths="source_path"
        ).with_columns(
            extract_game_id("source_path"),
            extract_period("source_path")
        )
    )
    return df if lazy else df.collect()

def read_id_mapping(path: str) -> Dict[str, str]:
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

def extract_game_id(col_name: str, file_pattern: str = '.*') -> pl.Expr:
    """
    Function to extract game_id from a column containing file paths.
    """
    return c(col_name).str.extract(rf"data/(\d+)/{file_pattern}").cast(pl.String).alias("game_id")

def extract_period(col_name: str) -> pl.Expr:
    """
    Function to extract period from a column containing file paths.
    """
    return c(col_name).str.extract(r"Period_(\d+)").cast(pl.Int32).alias("period")