from typing import Union, List

import polars as pl
from polars import col as c

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
    """
    Function to calculate total elapsed time since the start of the period, including stoppages. Resulting
    column can then be used to join with events via timecodes.
    """
    
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