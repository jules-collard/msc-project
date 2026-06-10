import json

import polars as pl
from polars import col as c

from parquet_helpers import EntityTrackingReader

def read_events(path: str, lazy=True) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to read event data from *_sapifullevents.json type files.
    """

    with open(path, "r") as f:
        events_dict = json.load(f)
    
    df = pl.from_dicts(events_dict.get("events"), infer_schema_length=None)
    return df.lazy() if lazy else df

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