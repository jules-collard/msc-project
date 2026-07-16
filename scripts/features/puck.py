import polars as pl
from polars import col as c
from polars import selectors as cs

from processing.tracking import adjust_vectors
from utils import distance_to_point_2d, magnitude_2d, distance_2d, project_y_to_goalline, project_z_to_goalline, cohens_kappa

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
        .pipe(adjust_vectors)
        .pipe(calculate_goal_vectors)
        .pipe(calculate_magnitudes)
        .with_columns(
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord').alias('dist_to_shot')
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

def evaluate_shot_detection(shots_with_features: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    shots_with_on_target = (
        shots_with_features
        .with_columns(
            on_target = (c('outcome') == 'successful'),
            est_on_target = (c('goalline_y').is_between(-3, 3) & c('goalline_z').is_between(0, 4))
        )
    )

    missing_data_summary = (
        shots_with_on_target
        .select(
            pl.any_horizontal(c('shot_time', 'shot_x', 'shot_y', 'shot_z').is_null()).mean().alias('shot_missing'),
            pl.any_horizontal(cs.starts_with('traj').is_null()).mean().alias('trajectory_missing'),
            pl.any_horizontal(cs.starts_with('goalline').is_null()).mean().alias('projection_missing')
        )
    )

    classification_summary = (
        shots_with_on_target
        # Only evaluate unblocked shots & shots with estimated on-target information
        .filter(
            c('type').str.contains('blocked').not_(),
            c('est_on_target').is_not_null()
        ).select(c('on_target', 'est_on_target'))
        .with_columns(
            true_positive = c('on_target') & c('est_on_target'),
            false_positive = c('on_target').not_() & c('est_on_target'),
            true_negative = c('on_target').not_() & c('est_on_target').not_(),
            false_negative = c('on_target') & c('est_on_target').not_()
        ).select(
            c('true_positive').sum().alias('true_positive'),
            c('false_positive').sum().alias('false_positive'),
            c('true_negative').sum().alias('true_negative'),
            c('false_negative').sum().alias('false_negative')
        ).with_columns(
            accuracy = (c('true_positive') + c('true_negative')) / pl.sum_horizontal(pl.all()),
            precision = c('true_positive') / (c('true_positive') + c('false_positive')),
            recall = c('true_positive') / (c('true_positive') + c('false_negative')),
            cohen_kappa = cohens_kappa('true_positive', 'true_negative', 'false_positive', 'false_negative')
        ).with_columns(
            f1_score = 2 * (c('precision') * c('recall')) / (c('precision') + c('recall'))
        ).drop(cs.starts_with('true', 'false'))
    )

    return pl.concat([missing_data_summary, classification_summary], how='horizontal')