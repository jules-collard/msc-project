import polars as pl
from polars import col as c

from processing.events import timecode_to_seconds, extract_flip
from processing.tracking import calculate_elapsed_time
from post_shot.detection import calculate_shot_detection
from utils import distance_to_point_2d

def shot_features() -> pl.Expr:
    """
    Returns a list of expressions with post-shot features.
    Should be applied to output of post_shot.detection.calculate_shot_detection()
    """

    on_goal = (c('goalline_y').abs() <= 3) & (c('goalline_z') <= 4)
    dist_to_post = 3 - c('goalline_y').abs()
    dist_to_crossbar = 4 - c('goalline_z')
    dist_to_top_corner = distance_to_point_2d(c('goalline_y').abs(), c('goalline_z'), 3, 4)
    dist_to_top_corner = ( # Distance to top corner is negative when not on target
        pl.when(on_goal)
        .then(dist_to_top_corner)
        .otherwise(-dist_to_top_corner)
    )
    dist_to_center = distance_to_point_2d(c('goalline_y'), c('goalline_z'), 0, 2)
    nearest_post_y = pl.when(c('goalline_y') < 0).then(-3).otherwise(3)

    return [
        on_goal.alias('on_goal'),
        dist_to_post.alias('dist_to_post'),
        dist_to_crossbar.alias('dist_to_crossbar'),
        dist_to_top_corner.alias('dist_to_top_corner'),
        dist_to_center.alias('dist_to_center'),
        nearest_post_y.alias('nearest_post_y')
    ]

def pipeline(
    events: pl.DataFrame | pl.LazyFrame,
    puck_tracking: pl.DataFrame | pl.LazyFrame,
    window_size: int = 1.6
) -> pl.DataFrame | pl.LazyFrame:
    """
    Full pipeline for post-shot model input. Takes in events and puck tracking data, without transformations
    (i.e. output of data_readers functions), and returns a dataframe with post-shot features.

    window_size: float, optional
        See post_shot.detection.calculate_shot_detection for details.
    """

    shots = (
        events
        .with_columns(
            extract_flip(),
            elapsed_time = timecode_to_seconds()
        )
        .filter(c('name') == 'shot')
        .sort(c('game_id', 'period', 'elapsed_time'))
        .with_row_index(name='shot_id')
    )

    puck_tracking = (
        puck_tracking
        .with_columns(elapsed_time = calculate_elapsed_time())
        .filter(
            c('entity_id') == '1'
        )
        .drop(c('entity_id', 'clock_state'))
    )
    
    shots_with_features = (
        calculate_shot_detection(shots, puck_tracking, window_size=window_size)
        .with_columns(shot_features())
        .drop_nulls(c('shot_speed'))
        .filter(
            c('shot_speed').is_not_null(), # Only shots where detection was successful
            c('shot_x') >= 25 # Only o-zone shots
        )
        .select(c('shot_id', 'shot_speed', 'on_goal', 'dist_to_post', 'dist_to_crossbar', 'dist_to_top_corner', 'dist_to_center'))
    )

    return (
        shots
        .select(c('shot_id'), c('expected_goals_all_shots').alias('pre_shot_xg'))
        .join(shots_with_features, on='shot_id', how='right')
    )