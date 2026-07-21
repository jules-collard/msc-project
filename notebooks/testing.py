import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    from polars import col as c

    from data_readers import batch_read_entity_tracking, batch_read_events, batch_read_puck_tracking, read_id_mapping
    from post_shot.detection import calculate_shot_detection
    from post_shot.features import PostShotData
    from post_shot.geometry import goal_vectors, project_vector
    from processing.tracking import calculate_elapsed_time, adjust_vectors
    from utils import distance_2d, magnitude_2d


@app.cell
def _():
    window_size: float = 1.6
    distance_threshold: float = 8
    impact_acceleration_threshold: float = -800
    deflection_angle_threshold: float = 25
    return (window_size,)


@app.cell
def _():
    events = batch_read_events("data/*/*_sapifullevents.json")

    puck_tracking = batch_read_puck_tracking(
        "data/*/HOCKEY_NHL_*_Period_*.parquet",
    )

    player_tracking = (
        batch_read_entity_tracking(
            "data/*/*_processed_measurements.parquet"
        )
    )

    mapping = read_id_mapping("data/NHL_20252026_player_sportlogiq_id_map.csv")
    return events, mapping, player_tracking, puck_tracking


@app.cell
def _(player_tracking):
    player_tracking.collect()
    return


@app.cell
def _(events, mapping, player_tracking, puck_tracking):
    post_shot_data = PostShotData(events=events, puck_tracking=puck_tracking, player_tracking=player_tracking, mapping=mapping)

    clean_puck_tracking = post_shot_data.puck_tracking_prepared
    shots = post_shot_data.shots
    clean_player_tracking = post_shot_data.player_tracking_prepared
    return clean_player_tracking, clean_puck_tracking, shots


@app.cell
def _(clean_player_tracking, clean_puck_tracking, shots, window_size: float):
    (
        clean_puck_tracking.sort(c('game_id', 'period', 'elapsed_time'))
        .join_asof(
            shots.sort(c('game_id', 'period', 'elapsed_time')),
            by=['game_id', 'period'],
            on='elapsed_time',
            strategy='nearest',
            tolerance=window_size / 2,
            coalesce=False
        ).drop_nulls(c('shot_id')) # Remove tracking outside of shot window
        .sort(c('game_id', 'period', 'entity_official_id', 'ts'))
        .join_asof( # Add shooter tracking data
            clean_player_tracking.sort(c('game_id', 'period', 'entity_official_id', 'ts')),
            by=['game_id', 'period', 'entity_official_id'],
            on='ts',
            strategy='backward',
            tolerance=0.15,
            coalesce=False,
            suffix='_player'
        ).drop_nulls(c('entity_id')) # Remove tracking outside of shot window
        .with_columns(adjust_vectors(c(
            'x', 'y', 'vx', 'vy', 'ax', 'ay',
            'x_player', 'y_player', 'vx_player', 'vy_player', 'ax_player', 'ay_player'
        ))).with_columns(
            goal_vectors()
        ).with_columns(
            project_vector('ax', 'ay', 'x_player_adj', 'y_player_adj', 'x_adj', 'y_adj').alias('acc_away_from_shooter'),
            speed = magnitude_2d('vx', 'vy'),
            acceleration = magnitude_2d('ax', 'ay'),
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord'),
            dist_to_shooter = distance_2d('x_adj', 'y_adj', 'x_player_adj', 'y_player_adj')
        ).with_columns(
            angle_vel = c('angle_to_goal').diff().over(c('shot_id'))
        ).collect().select(c('dist_to_shooter'))
    )
    return


@app.cell
def _(clean_player_tracking, clean_puck_tracking, shots):
    calculate_shot_detection(shots, clean_puck_tracking, clean_player_tracking).collect()
    return


if __name__ == "__main__":
    app.run()
