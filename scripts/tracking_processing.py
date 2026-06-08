import polars as pl
from polars import col as c

def derive_game_clock(tracking: pl.DataFrame | pl.LazyFrame): # USE FOR SINGLE GAME ONLY
    """
    Function to add period_time and game_time fields to tracking data.
    
    Uses the raw timestamps and clock_state information (1=moving, 2=stopped) to derive game clock.
    Game time calculation is reset at each period to avoid build-up of timing errors
    """
    return (
        tracking
        .sort(c('ts'))
        .with_columns(delta = c('ts').diff())
        # Don't increment game clock when clock is stopped
        .with_columns(delta = pl.when(c('clock_state') == 1).then(c('delta')).otherwise(0).fill_null(0))
        .with_columns(
            period_time = c('delta').cum_sum().over('period')
        ).with_columns(
            game_time = (c('period') - 1) * 1200 + c('period_time')
        )
    )

def convert_timestamps(tracking: pl.DataFrame | pl.LazyFrame):
    """
    Function to convert ts field from UNIX epochs to DateTime format (only for legibility)
    """
    return tracking.with_columns(pl.from_epoch(c("ts"), time_unit="s").alias("ts"))