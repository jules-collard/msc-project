import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_line, geom_vline

    from data_readers import read_events, read_puck_tracking
    from tracking_processing import derive_game_clock, adjust_vectors, calculate_goal_vectors, calculate_magnitudes
    from event_processing import add_flip

    from utils import distance_2d


@app.cell
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json").pipe(add_flip)
    shots = events.filter(c('name') == 'shot').sort(c('game_time')).with_row_index(name='shot_id')
    return (shots,)


@app.cell
def _():
    periods = [1,2,3]
    puck_tracking = read_puck_tracking(
        [f"data/one_game/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_{i}.json" for i in periods], periods
    ).pipe(derive_game_clock).filter(c('clock_state') == 1).sort(c('game_time'))
    return (puck_tracking,)


@app.cell
def _(puck_tracking, shots):
    tracking_with_shots = (
        puck_tracking
        .join_asof(
            shots,
            on='game_time',
            strategy='nearest',
            tolerance=0.5,
        ).drop_nulls(c('shot_id'))
        .pipe(adjust_vectors)
        .pipe(calculate_goal_vectors)
        .pipe(calculate_magnitudes)
        .with_columns(
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord').alias('dist_to_shot')
        ).collect()
    )
    return (tracking_with_shots,)


@app.cell
def _(shots):
    max_id = shots.select(c('shot_id').max()).collect().item()

    id_selector = mo.ui.number(start=0, stop=max_id, step=1)
    return (id_selector,)


@app.cell
def _(id_selector, shots, tracking_with_shots):
    sample = tracking_with_shots.filter(c('shot_id') == id_selector.value)
    game_time = shots.filter(c('shot_id') == id_selector.value).select(c('game_time')).collect().item()
    return game_time, sample


@app.cell
def _(shots):
    shots.collect()
    return


@app.cell
def _(game_time, id_selector, sample):
    custom_theme = (
        p9.theme_bw()
        + p9.theme(
            axis_title_x=p9.element_blank(),
            axis_text_x=p9.element_text(angle=-90)
        )
    )

    speed_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='speed'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + custom_theme
    )

    goal_speed_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='goal_speed'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + custom_theme
    )

    acc_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='acceleration'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + custom_theme
    )

    goal_acc_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='goal_acceleration'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + p9.geom_hline(yintercept=0, color='grey')
        + custom_theme
    )

    angle_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='angle_to_goal'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + custom_theme
        + p9.scale_y_continuous(limits=(-180, 180), breaks=[-180, -90, 0, 90, 180])
    )

    distance_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='dist_to_shot'))
        + geom_vline(xintercept=game_time, linetype='dashed', color='red')
        + custom_theme
        + p9.ylim(0, None)
    )


    mo.vstack([
        id_selector,
        (speed_plot | acc_plot | angle_plot) / (goal_speed_plot | goal_acc_plot | distance_plot)
    ])
    return


if __name__ == "__main__":
    app.run()
