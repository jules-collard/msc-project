import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_line

    from data_readers import read_events, read_puck_tracking
    from tracking_processing import derive_game_clock, adjust_vectors, calculate_goal_vectors
    from event_processing import add_flip


@app.cell
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json").pipe(add_flip)
    shots = events.filter(c('name') == 'shot').sort(c('game_time')).with_row_index(name='shot_id').pipe(add_flip)
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
    GOAL_X = 89
    GOAL_Y = 0

    tracking_with_shots = (
        puck_tracking
        .join_asof(
            shots,
            on='game_time',
            strategy='nearest',
            tolerance=0.4,
        ).drop_nulls(c('shot_id'))
        .pipe(adjust_vectors)
        .pipe(calculate_goal_vectors)
        .collect()
    )
    return (tracking_with_shots,)


@app.cell
def _(shots):
    max_id = shots.select(c('shot_id').max()).collect().item()

    id_selector = mo.ui.number(start=0, stop=max_id, step=1)
    return (id_selector,)


@app.cell
def _(id_selector, tracking_with_shots):
    sample = tracking_with_shots.filter(c('shot_id') == id_selector.value)
    return (sample,)


@app.cell
def _(id_selector, sample):
    speed_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='goal_speed'))
    )

    acc_plot = (
        ggplot(sample)
        + geom_line(aes(x='game_time', y='goal_acceleration'))
    )

    mo.vstack([
        id_selector,
        speed_plot | acc_plot
    ])
    return


if __name__ == "__main__":
    app.run()
