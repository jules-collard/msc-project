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
    from post_shot.features import PostShotData
    from interaction import game_selectors, display_game_selectors


@app.cell(hide_code=True)
def _():
    game_id_mapping = (
        pl.read_csv("mappings/NHL_20252026_game_smt_sportlogiq_id_map.csv")
        .with_columns(c('GameDate').str.strptime(pl.Date, format="%Y-%m-%d"))
    )
    return (game_id_mapping,)


@app.cell(hide_code=True)
def _(game_id_mapping):
    date_selector, game_id_selector, game_type_selector, run_button = game_selectors(game_id_mapping)
    display_game_selectors(date_selector, game_id_selector, game_type_selector, run_button)
    return date_selector, game_id_selector, game_type_selector, run_button


@app.cell
def _(date_selector, game_id_mapping, game_id_selector, game_type_selector):
    games = (
        game_id_mapping
        .filter(
            c('GameDate').is_in(pl.date_range(*date_selector.value).implode()),
            c('SportlogiqGameID').is_in(game_id_selector.value),
            c('Stage').is_in(game_type_selector.value)
        )
    )

    sportlogiq_ids = games.select(c('SportlogiqGameID')).to_series().to_list()
    SMT_ids = games.select(c('SMTGameID')).to_series().to_list()
    return SMT_ids, games, sportlogiq_ids


@app.cell
def _(SMT_ids, games, run_button, sportlogiq_ids):
    mo.stop(not run_button.value, mo.md("Press **Run Shot Detection** above to load the data and generate the plots."))

    events = batch_read_events([f"data/sportlogiq/*/games/{id}/*_sapifullevents.json" for id in sportlogiq_ids])

    puck_tracking = batch_read_puck_tracking(
        [f"data/smtoasis/*/games/{id}/*_puck_tracking_raw_measurements*.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )
    return events, puck_tracking


@app.cell
def _():
    distance_slider = mo.ui.slider(start=2, stop=15, step=1, value=8, debounce=True, include_input=True, label="Distance Threshold (ft)")
    acc_slider = mo.ui.slider(start=-2000, stop=0, step=100, value=-800, debounce=True, include_input=True, label="Impact Acceleration Threshold (ft/s²)")
    angle_slider = mo.ui.slider(start=0, stop=100, step=5, value=25, debounce=True, include_input=True, label="Angular Velocity Threshold (degrees/s)")
    return acc_slider, angle_slider, distance_slider


@app.cell
def _(
    acc_slider,
    angle_slider,
    distance_slider,
    events,
    puck_tracking,
    run_button,
):
    post_shot_data = PostShotData(events, puck_tracking, distance_threshold=distance_slider.value, impact_acceleration_threshold=acc_slider.value, deflection_angle_threshold=angle_slider.value)
    metrics = post_shot_data.evaluate_detection().collect()
    return metrics, post_shot_data


@app.cell
def _(acc_slider, angle_slider, distance_slider, metrics):
    mo.vstack([distance_slider, acc_slider, angle_slider, mo.ui.table(metrics)])
    return


@app.cell
def _(post_shot_data):
    full_output = post_shot_data.full_output().collect()
    return (full_output,)


@app.cell(hide_code=True)
def _(full_output):
    timing_plot = (
        full_output
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
def _(full_output):
    location_plot = (
        full_output
        .with_columns(
            x_error = c('x_adj_coord') - c('shot_x'),
            y_error = c('y_adj_coord') - c('shot_y')
        )
        >> ggplot(aes(x='x_error', y='y_error'))
        + geom_vline(xintercept=0, alpha=0.8)
        + geom_hline(yintercept=0, alpha=0.8)
        + p9.geom_point(alpha=0.1)
        + p9.stat_density_2d(aes(fill='stat(level)'), geom='polygon', levels=10, alpha=0.6)
        + p9.geom_density_2d(levels=10, alpha=0.5)
        + p9.geom_rug()
        + p9.scale_fill_distiller(type='seq', palette='Blues', direction=1)
        + theme_bw(base_size=10)
        + labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)", title="Distribution of Estimated Shot Location Errors", fill="Density", caption="X/Y Coordinate Error = Event Shot Location - Estimated Shot Location")
        + p9.theme(aspect_ratio=1)
    )

    # location_plot.save("plots/shot_detection/location_error_distribution.svg")

    location_plot
    return


if __name__ == "__main__":
    app.run()
