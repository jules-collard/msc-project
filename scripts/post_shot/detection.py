import polars as pl
from polars import col as c
from polars import selectors as cs

from post_shot.geometry import goal_vectors, project_y_to_goalline, project_z_to_goalline
from processing.tracking import adjust_vectors
from utils import magnitude_2d, distance_2d

def calculate_shot_detection(
        shots: pl.DataFrame | pl.LazyFrame,
        puck_tracking: pl.DataFrame | pl.LazyFrame,
        window_size: float = 1.6,
        distance_threshold: float = 8,
        impact_acceleration_threshold: float = -800,
        deflection_angle_threshold: float = 25
) -> pl.DataFrame | pl.LazyFrame:
    """
    Calculates puck tracking features for each shot. Shots dataframe should be events-type dataframe, filtered
    to only include shots, with a 'shot_id' column, which must be unique across ENTIRE (loaded) dataset. Tracking
    dataframe should only contain puck tracking data, with game clock derived.

    window_size: float, optional
        The size of the window (in seconds) to select tracking data for each shot. Default is 1.6 seconds.
    distance_threshold: float, optional
        The maximum distance (in feet) from the shot location to consider tracking data for the shot.
    impact_acceleration_threshold: float, optional
        The threshold for the goal acceleration to consider the shot as having been blocked/saved/deflected.
    deflection_angle_threshold: float, optional
        The threshold for the change in angle to consider the shot as having been deflected.
    """

    # Add shot details to corresponding puck tracking data, and calculate features
    tracking_with_shots = (
        puck_tracking.sort(c('game_id', 'period', 'elapsed_time'))
        .join_asof(
            shots.sort(c('game_id', 'period', 'elapsed_time')),
            by=['game_id', 'period'],
            on='elapsed_time',
            strategy='nearest',
            tolerance=window_size / 2,
            coalesce=False
        ).drop_nulls(c('shot_id'))
        .with_columns(adjust_vectors(c('x', 'y', 'vx', 'vy', 'ax', 'ay')))
        .with_columns(
            goal_vectors(),
            speed = magnitude_2d('vx', 'vy'),
            acceleration = magnitude_2d('ax', 'ay'),
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord')
        ).with_columns(
            angle_vel = c('angle_to_goal').diff().over(c('shot_id'))
        )
    )

    return (
        tracking_with_shots
        .sort(c('game_id', 'period', 'shot_id', 'elapsed_time'))
        .with_columns(
            pl.int_range(pl.len()).over(c('shot_id')).alias('frame_index')
        ).with_columns( # Conditions for valid shot frames
            valid_shot_frame = (c('dist_to_shot') <= distance_threshold) 
            & (c('angle_to_goal').abs() <= 90)
            & (c('goal_speed') > 0)
            & (c('goal_acceleration') > 0)
        ).with_columns(
            masked_acceleration = pl.when(c('valid_shot_frame')).then(c('acceleration')).otherwise(None)
        ).with_columns( # Identify frame where shot occurs
            shot_frame = c('masked_acceleration').arg_max().over(c('shot_id')),
        ).with_columns( # Conditions for stopping the shot trajectory
            impact_condition = (c('goal_acceleration') < impact_acceleration_threshold),
            deflection_condition = (c('angle_vel').abs() > deflection_angle_threshold),
            goal_line_condition = ((c('x_adj') >= 89) & (c('x_adj_coord') < 89)) | ((c('x_adj') < 89) & (c('x_adj_coord') >= 89))
        ).with_columns(
            stop = ((c('frame_index') > c('shot_frame') + 5) & pl.any_horizontal(cs.ends_with('condition'))).over(c('shot_id'))
        ).with_columns( # Take first frame where stop condition is met, otherwise take last frame of shot trajectory
            (
                c('stop').arg_true().first()
                .fill_null(c('frame_index').filter(c('frame_index') > c('shot_frame')).last())
                .over(c('shot_id'))
                .alias('stop_frame')
            )
        ).with_columns(
            masked_speed = pl.when(
                # Only consider speed within shot window
                c('frame_index') >= c('shot_frame'),
                c('frame_index') <= c('stop_frame')
            ).then(c('speed')).otherwise(None),
        ).with_columns( # Identify frame where max shot speed occurs
            speed_frame = c('masked_speed').arg_max().over(c('shot_id')),
        ).group_by(c('shot_id'))
        .agg(
            c('elapsed_time').filter(c('frame_index') == c('shot_frame')).first().alias('shot_time'),
            c('elapsed_time').filter(c('frame_index') < c('stop_frame')).last().alias('shot_end_time'),
            c('elapsed_time_right').filter(c('frame_index') == c('shot_frame')).first().alias('event_time'),
            c('speed').filter(c('frame_index') == c('speed_frame')).first().alias('shot_speed'),
            c('x_adj').filter(c('frame_index') == c('shot_frame')).first().alias('shot_x'),
            c('y_adj').filter(c('frame_index') == c('shot_frame')).first().alias('shot_y'),
            c('z').filter(c('frame_index') == c('shot_frame')).first().alias('shot_z'),
            c('elapsed_time').filter(c('frame_index') < c('stop_frame')).last().alias('traj_time'),
            c('x_adj').filter(c('frame_index') < c('stop_frame')).last().alias('traj_x'),
            c('y_adj').filter(c('frame_index') < c('stop_frame')).last().alias('traj_y'),
            c('z').filter(c('frame_index') < c('stop_frame')).last().alias('traj_z'),
        ).with_columns(
            project_y_to_goalline('shot_x', 'shot_y', 'traj_x', 'traj_y'),
            project_z_to_goalline('shot_x', 'shot_z', 'shot_time', 'traj_x', 'traj_z', 'traj_time')
        )
    )