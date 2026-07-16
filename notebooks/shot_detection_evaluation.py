import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, labs, geom_vline, geom_hline, theme_bw

    from data_readers import batch_read_events, batch_read_puck_tracking
    from processing.tracking import calculate_elapsed_time
    from processing.events import extract_flip, timecode_to_seconds
    from features.puck import calculate_shot_detection, evaluate_shot_detection


@app.cell
def _():
    game_id_selector = mo.ui.text(placeholder="Enter Game ID", value="*")
    game_id_selector
    return (game_id_selector,)


@app.cell(hide_code=True)
def _(game_id_selector):
    events = batch_read_events(f"data/{game_id_selector.value}/*_sapifullevents.json").with_columns(extract_flip())
    shots = (
        events
        .with_columns(elapsed_time = timecode_to_seconds())
        .filter(c('name') == 'shot')
        .sort(c('game_id', 'period', 'elapsed_time'))
        .with_row_index(name='shot_id')
    )

    puck_tracking = batch_read_puck_tracking(
        f"data/{game_id_selector.value}/HOCKEY_NHL_*_Period_*.parquet",
    )

    puck_tracking = (
        puck_tracking
        .with_columns(elapsed_time = calculate_elapsed_time())
        .filter(c('entity_id') == '1')
        .sort(c('game_id', 'period', 'elapsed_time'))
        .drop(c('entity_id', 'clock_state'))
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
    mo.vstack([distance_slider, acc_slider, angle_slider, mo.ui.table(metrics)])
    return


@app.cell(hide_code=True)
def _(shots_with_features):
    timing_plot = (
        shots_with_features
        .with_columns(timing_error = c('elapsed_time') - c('shot_time'))
        >> ggplot(aes(x='timing_error'))
        + geom_vline(xintercept=0, linetype='dashed')
        + p9.geom_density(fill='lightblue', alpha=0.5)
        # + geom_histogram(bins=25, color='black')
        + p9.theme_bw(base_size=10)
        + p9.xlim((-0.5, 0.5))
        + labs(x="Estimated Timing Error (seconds)", y="Density", title="Distribution of Estimated Timing Errors for Shots",
              caption="Timing Error = Event Time - Estimated Shot Time")
    )

    # timing_plot.save("plots/shot_detection/timing_error_distribution.svg")

    timing_plot
    return


@app.cell(hide_code=True)
def _(shots_with_features):
    location_plot = (
        shots_with_features
        .with_columns(
            x_error = c('x_adj_coord') - c('shot_x'),
            y_error = c('y_adj_coord') - c('shot_y')
        )
        >> ggplot(aes(x='x_error', y='y_error'))
        + geom_vline(xintercept=0, alpha=0.8)
        + geom_hline(yintercept=0, alpha=0.8)
        + p9.stat_density_2d(aes(fill='stat(level)'), geom='polygon', levels=10, alpha=0.6)
        + p9.geom_density_2d(levels=10, alpha=0.5)
        + p9.geom_rug()
        + p9.scale_fill_distiller(type='seq', palette='Blues', direction=1)
        + theme_bw(base_size=10)
        + labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)", title="Distribution of Estimated Shot Location Errors", fill="Density", caption="X/Y Coordinate Error = Event Shot Location - Estimated Shot Location")
    )

    # location_plot.save("plots/shot_detection/location_error_distribution.svg")

    location_plot
    return


if __name__ == "__main__":
    app.run()
