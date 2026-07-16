from typing import Union, List

import polars as pl
from polars import col as c

from utils import distance_to_point_2d, magnitude_2d

def derive_game_clock(tracking: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame: # USE FOR SINGLE GAME ONLY
    """
    Function to add period_time and game_time fields to tracking data.
    
    Uses the raw timestamps and clock_state information (1=moving, 2=stopped) to derive game clock.
    Game time calculation is reset at each period to avoid build-up of timing errors
    """
    return (
        tracking
        .sort(c('ts'))
        .with_columns(delta = c('ts').diff().over('game_id', 'period'))
        # Don't increment game clock when clock is stopped
        .with_columns(delta = pl.when(c('clock_state') == 1).then(c('delta')).otherwise(0).fill_null(0))
        .with_columns(
            period_time = c('delta').cum_sum().over('game_id', 'period')
        ).with_columns(
            game_time = (c('period') - 1) * 1200 + c('period_time')
        ).drop(c('delta'))
    )

def convert_timestamps(tracking: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to convert ts field from UNIX epochs to DateTime format (only for legibility)
    """
    return tracking.with_columns(pl.from_epoch(c("ts"), time_unit="s").alias("ts"))

def adjust_vectors(tracking: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Adds adj_... columns to tracking locations, velocities and accelerations to correct
    for attacking direction.
    """

    if isinstance(tracking, pl.DataFrame):
        assert 'flip' in tracking.columns
    elif isinstance(tracking, pl.LazyFrame):
        assert 'flip' in tracking.collect_schema().names()
    else:
        raise TypeError

    return (
        tracking
        .with_columns(
            pl.when(c('flip'))
            .then(-c('x', 'y', 'vx', 'vy', 'ax', 'ay'))
            .otherwise(c('x', 'y', 'vx', 'vy', 'ax', 'ay'))
            .name.suffix('_adj')
        )
    )

def calculate_goal_vectors(tracking: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    GOAL_X = 89
    GOAL_Y = 0
    
    return (
        tracking
        .with_columns(
            # 1. Calculate the vector from the puck to the net
            (GOAL_X - pl.col("x_adj")).alias("dx_to_goal"),
            (GOAL_Y - pl.col("y_adj")).alias("dy_to_goal")
        ).with_columns(
            # 2. Calculate the distance (magnitude of the vector)
            (distance_to_point_2d("x_adj", "y_adj", GOAL_X, GOAL_Y) + 1e-6).alias("dist_to_goal")
        ).with_columns(
            # 3. Calculate Unit Vector components (normalized direction)
            (pl.col("dx_to_goal") / pl.col("dist_to_goal")).alias("u_x"),
            (pl.col("dy_to_goal") / pl.col("dist_to_goal")).alias("u_y")
        ).with_columns(
            # Dot Products: Project velocity and acceleration onto the unit vector
            ((pl.col("vx_adj") * pl.col("u_x")) + (pl.col("vy_adj") * pl.col("u_y"))).alias("goal_speed"),
            ((pl.col("ax_adj") * pl.col("u_x")) + (pl.col("ay_adj") * pl.col("u_y"))).alias("goal_acceleration"),
            # Cross Product
            ((pl.col("u_x") * pl.col("vy_adj")) - (pl.col("u_y") * pl.col("vx_adj"))).alias("tangent_speed")
        ).with_columns(
            pl.arctan2(pl.col("tangent_speed"), pl.col("goal_speed")).degrees().alias("angle_to_goal")
        ).drop(
            "dx_to_goal", "dy_to_goal", "dist_to_goal", "u_x", "u_y", "tangent_speed"
        )
    )

def calculate_magnitudes(tracking: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return (
        tracking.with_columns(
            speed = magnitude_2d('vx', 'vy'),
            acceleration = magnitude_2d('ax', 'ay'),
        )
    )

def calculate_elapsed_time(
    ts: Union[str, pl.Expr] = 'ts', 
    clock_state: Union[str, pl.Expr] = 'clock_state', 
    over_cols: List[Union[str, pl.Expr]] = ['game_id', 'period']
) -> pl.Expr:
    
    # Ensure inputs are expressions
    ts_expr = pl.col(ts) if isinstance(ts, str) else ts
    clock_expr = pl.col(clock_state) if isinstance(clock_state, str) else clock_state
    
    # Calculate the minimum timestamp for the period where the clock is running
    period_start_ts = (
        pl.when(clock_expr == 1)
        .then(ts_expr)
        .otherwise(None)
        .min()
        .over(over_cols)
    )
    
    # Subtract the period start from the current timestamp
    return ts_expr - period_start_ts