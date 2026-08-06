from typing import Literal

import polars as pl

Framework = Literal['xgboost', 'lightgbm', 'xgboost-dart', 'lightgbm-dart']
ImbalanceStrategy = Literal['RO', 'RU', 'WCE']

ShotType = pl.Enum(["wristshot", "slapshot", "snapshot", "backhand", "forehandbackhand", "backhandforehand", "wraparound", "deflected"])
Handedness = pl.Enum(["L", "R"])