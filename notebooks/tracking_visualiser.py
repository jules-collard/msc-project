import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    import numpy as np
    from hockey_rink import NHLRink
    from matplotlib import pyplot as plt

    from processing.tracking import derive_game_clock
    from data_readers import read_entity_registration, read_entity_tracking, read_events, batch_read_puck_tracking
    from processing.events import remove_non_viz_events


@app.cell
def _():
    player_tracking = (
        read_entity_tracking("data/20260521/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet")
    )

    puck_tracking = batch_read_puck_tracking(
        "data/20260521/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_*.parquet"
    )

    rosters = read_entity_registration("data/20260521/NHL_20252026_postseason_20260521_MTLvsCAR_entity_registration.json", lazy=False)
    events = read_events("data/20260521/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json", lazy=False)
    return events, player_tracking, puck_tracking, rosters


@app.cell
def _(player_tracking, puck_tracking):
    tracking = pl.concat([player_tracking, puck_tracking], how='diagonal_relaxed').pipe(derive_game_clock).filter(c('clock_state') == 1).sort(c('game_time')).collect()
    return (tracking,)


@app.cell
def _(events):
    events_viz = (
        events
        .pipe(remove_non_viz_events)
    )
    return (events_viz,)


@app.cell
def _(rosters, tracking):
    entity_ids = tracking.select(c('entity_id')).unique()
    timestamps = pl.from_dict(
        {'period': np.repeat(np.arange(1, 4), 1200 / 0.1), 'period_time': np.tile(np.arange(1200, step=0.1), 3).round(1)},
        schema={'period': pl.Int32(), 'period_time': pl.Float64()}
    ).with_columns(
        game_time=(c('period') - 1) * 1200 + c('period_time')
    )

    tracking_at_time = ( 
        timestamps
        .join(entity_ids, how='cross')
        .sort(c('game_time'))
        .join(rosters, left_on='entity_id', right_on='EntityId', how='left')
        .join_asof(
            tracking.sort(c('game_time')),
            on=c('game_time'), 
            by='entity_id', 
            strategy='forward', 
            tolerance=0.1
        ).filter(
            c('entity_official_id').is_not_null().or_(c('entity_id') == '1')
        ) # Remove players not on ice
    )
    return (tracking_at_time,)


@app.cell
def _(events_viz, period_selector, time_selector):
    current_event_df = (
        events_viz
        .filter(c('period') == period_selector.value)
        .with_columns(time_diff = (c('period_time') - time_selector.value).abs())
        .filter(c('time_diff') <= 0.1)
        .head(1)
    )

    if not current_event_df.is_empty():
        current_event = current_event_df.to_dicts()[0]
    else:
        current_event = None
    return (current_event,)


@app.cell
def _():
    period_selector = mo.ui.number(start=1, stop=3, step=1)
    time_selector = mo.ui.number(start=0, stop=1200, step=0.1)
    return period_selector, time_selector


@app.cell
def _(current_event, period_selector, time_selector, tracking_at_time):
    display_tracking = tracking_at_time.filter(
        c('entity_id') != '1',
        c('period') == period_selector.value,
        c('period_time') == time_selector.value
    )
    puck_display_tracking = tracking_at_time.filter(
        c('entity_id') == '1',
        c('period') == period_selector.value,
        c('period_time') == time_selector.value
    )

    rink = NHLRink()

    rink.arrow(
        x=display_tracking.select(c('x')), 
        y=display_tracking.select(c('y')),
        dx=display_tracking.select(c('vx')), 
        dy=display_tracking.select(c('vy'))
    )

    rink.scatter(
        x=display_tracking.select(c('x')), 
        y=display_tracking.select(c('y')),
        c=display_tracking.select(c('VisOrHome').replace({'Home': 'tab:red', 'Visitor': 'tab:blue'})).to_series(),
        s=250, 
        edgecolor='black'
    )

    rink.text(
        display_tracking.select(c('x')), 
        display_tracking.select(c('y')), 
        display_tracking.select(c('JerseyNum')), 
        fontsize=10, 
        ha="center", 
        va="center", 
        color="white"
    )

    rink.scatter(
        x=puck_display_tracking.select(c('x')),
        y=puck_display_tracking.select(c('y')),
        s=100, 
        c='black'
    )

    if current_event:
        rink.scatter(
            x=current_event['x_coord'],
            y=current_event['y_coord'], 
            s=100, 
            c='black', 
            marker='x'
        )
        plt.title(f"{current_event['name']} - {current_event['shorthand']}: {current_event['outcome']} ({current_event['flags']})")

    mo.vstack([
        mo.hstack([period_selector, time_selector], justify='start'),
        rink.draw()
    ])

    # plt.savefig("example_tracking.svg", transparent=True, bbox_inches="tight", format="svg")
    return


if __name__ == "__main__":
    app.run()
