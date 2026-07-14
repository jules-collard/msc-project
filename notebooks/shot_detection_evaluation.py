import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, labs, geom_histogram, geom_vline, geom_hline, theme_bw

    from data_readers import read_events, read_entity_tracking, read_puck_tracking
    from tracking_processing import derive_game_clock
    from event_processing import add_flip
    from features.puck import calculate_shot_detection, evaluate_shot_detection
    from utils import cohens_kappa


@app.cell(hide_code=True)
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json").pipe(add_flip)
    shots = events.filter(c('name') == 'shot').sort(c('game_time')).with_row_index(name='shot_id')

    periods = [1,2,3]
    player_tracking = read_entity_tracking(
        "data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet"
    )

    puck_tracking = read_puck_tracking(
        [f"data/one_game/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_{i}.json" for i in periods], periods
    )

    puck_tracking = (
        pl.concat([player_tracking, puck_tracking], how='diagonal_relaxed')
        .pipe(derive_game_clock)
        .filter(c('clock_state') == 1, c('entity_id') == '1')
        .sort(c('game_time'))
        .drop(c('entity_id', 'entity_official_id', 'segment_idx', 'clock_state', 'raw_x', 'raw_y', 'raw_z'))
    )
    return puck_tracking, shots


@app.cell(hide_code=True)
def _(acc_slider, angle_slider, distance_slider, puck_tracking, shots):
    shot_features = calculate_shot_detection(shots, puck_tracking, distance_threshold=distance_slider.value, impact_acceleration_threshold=acc_slider.value, deflection_angle_threshold=angle_slider.value)
    shots_with_features = shots.join(shot_features, on=c('shot_id'), how='left').collect()
    return (shots_with_features,)


@app.cell
def _():
    distance_slider = mo.ui.slider(start=2, stop=15, step=1, value=8, debounce=True, include_input=True, label="Distance Threshold (ft)")
    acc_slider = mo.ui.slider(start=-2000, stop=0, step=100, value=-800, debounce=True, include_input=True, label="Impact Acceleration Threshold (ft/s²)")
    angle_slider = mo.ui.slider(start=0, stop=100, step=5, value=25, debounce=True, include_input=True, label="Angular Velocity Threshold (degrees/s)")
    return acc_slider, angle_slider, distance_slider


@app.cell
def _(shots_with_features):
    metrics = evaluate_shot_detection(shots_with_features)
    return (metrics,)


@app.cell
def _(acc_slider, angle_slider, distance_slider, metrics):
    mo.vstack([distance_slider, acc_slider, angle_slider, metrics]
    )
    return


@app.cell(hide_code=True)
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(timing_error = c('game_time') - c('shot_time'))
        >> ggplot(aes(x='timing_error'))
        + geom_vline(xintercept=0, linetype='dashed')
        + p9.geom_density(fill='lightblue', alpha=0.5)
        # + geom_histogram(bins=25, color='black')
        + p9.theme_bw(base_size=10)
        + labs(x="Estimated Timing Error (seconds)", y="Density", title="Distribution of Estimated Timing Errors for Shots",
              caption="Timing Error = Event Time - Estimated Shot Time")
    )
    return


@app.cell(hide_code=True)
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(
            x_error = c('x_adj_coord') - c('shot_x'),
            y_error = c('y_adj_coord') - c('shot_y')
        )
        >> ggplot(aes(x='x_error', y='y_error'))
        + geom_vline(xintercept=0, alpha=0.8)
        + geom_hline(yintercept=0, alpha=0.8)
        + p9.geom_density_2d(aes(color='..level..'), levels=9)
        + p9.geom_rug()
        + p9.scale_color_distiller(type='seq', palette='Oranges', direction=1)
        + theme_bw(base_size=10)
        + labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)", title="2D Density of Estimated Shot Location Errors", color="Density", caption="X/Y Coordinate Error = Event Shot Location - Estimated Shot Location")
    )
    return


if __name__ == "__main__":
    app.run()
