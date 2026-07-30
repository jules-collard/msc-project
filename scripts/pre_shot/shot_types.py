import polars as pl

ShotTypesEnum = pl.Enum([
    "wristshot",
    "slapshot",
    "snapshot",
    "backhand",
    "forehandbackhand",
    "backhandforehand",
    "wraparound",
    "deflected"
])