from typing import Union, List

import polars as pl
from polars import col as c

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

def convert_timestamps(expr: pl.Expr | str) -> pl.Expr:
    """
    Function to convert ts field from UNIX epochs to DateTime format (only for legibility)
    """

    expr = c(expr) if isinstance(expr, str) else expr
    return pl.from_epoch(expr, time_unit="s")

def adjust_vectors(expr: pl.Expr, flip_expr: pl.Expr = c('flip')) -> pl.Expr:
    """
    Function to adjust vectors (specified by input expression) for attacking direction.
    """
    return (
        pl.when(flip_expr)
        .then(-expr)
        .otherwise(expr)
        .name.suffix('_adj')
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
    return (ts_expr - period_start_ts).alias('elapsed_time')