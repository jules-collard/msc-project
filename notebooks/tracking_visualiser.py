import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import json

    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import numpy as np
    from hockey_rink import NHLRink
    from matplotlib import pyplot as plt

    from parquet_helpers import EntityTrackingReader
    from tracking_processing import derive_game_clock
    from event_processing import add_pass_target, add_carry_info, add_shot_info, remove_non_viz_events


@app.cell
def _():
    with open("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet", mode="rb") as f:
        reader = EntityTrackingReader(f.read())

    tracking_pa = reader.get_table()
    tracking = pl.from_arrow(tracking_pa).lazy().pipe(derive_game_clock).collect()
    return (tracking,)


@app.cell
def _():
    with open("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_registration.json") as roster_f:
        rosters_json = json.load(roster_f)

    rosters = pl.from_dicts(rosters_json[0].get("Entities"))
    return (rosters,)


@app.cell
def _():
    with open("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json", "r") as file:
        events_dict = json.load(file)

    events = pl.from_dicts(events_dict.get("events"), infer_schema_length=None)
    return (events,)


@app.cell
def _(events):
    events_viz = (
        events
        .pipe(add_pass_target)
        .pipe(remove_non_viz_events)
        .pipe(add_carry_info)
    )
    return (events_viz,)


@app.cell
def _(events_viz):
    events_viz
    return


@app.cell
def _(rosters, tracking):
    player_ids = tracking.select(c('entity_official_id')).unique()
    timestamps = pl.from_dict(
        {'period': np.repeat(np.arange(1, 4), 1200 / 0.1), 'period_time': np.tile(np.arange(1200, step=0.1), 3).round(1)},
        schema={'period': pl.Int32(), 'period_time': pl.Float64()}
    ).with_columns(
        game_time=(c('period') - 1) * 1200 + c('period_time')
    )

    tracking_at_time = ( 
        timestamps
        .join(player_ids, how='cross')
        .sort(c('game_time'))
        .join(rosters, left_on='entity_official_id', right_on='OfficialId', how='left')
        .join_asof(
            tracking.sort(c('game_time')),
            on=c('game_time'), 
            by='entity_official_id', 
            strategy='forward', 
            tolerance=0.1
        ).drop_nulls(c('entity_id')) # Remove players not on ice
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
    display_tracking = tracking_at_time.filter(c('period') == period_selector.value, c('period_time') == time_selector.value)

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

    if current_event:
        rink.scatter(
            x=current_event['x_coord'],
            y=current_event['y_coord'], 
            s=100, 
            c='black'
        )
        plt.title(f"{current_event['name']} - {current_event['shorthand']}: {current_event['outcome']} ({current_event['flags']})")
        if current_event['name'] == 'pass':
            rink.wavy_arrow(
                x=current_event['x_coord'],
                y=current_event['y_coord'],
                x2=current_event['pass_target_x_coord'], 
                y2=current_event['pass_target_y_coord'],
                zorder=5
            )

    mo.vstack([
        mo.hstack([period_selector, time_selector], justify='start'),
        rink.draw()
    ])
    return


if __name__ == "__main__":
    app.run()
