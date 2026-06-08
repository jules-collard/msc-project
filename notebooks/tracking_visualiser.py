import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import json

    import marimo as mo
    import polars as pl
    from polars import col as c
    import numpy as np
    from hockey_rink import NHLRink

    from parquet_helpers import EntityTrackingReader
    from tracking_processing import derive_game_clock


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
        ).drop_nulls(c('entity_id'))
    )
    return (tracking_at_time,)


@app.cell
def _():
    period_selector = mo.ui.number(start=1, stop=3, step=1)
    time_selector = mo.ui.number(start=0, stop=1200, step=0.1)
    return period_selector, time_selector


@app.cell
def _(period_selector, time_selector, tracking_at_time):
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

    mo.vstack([
        mo.hstack([period_selector, time_selector], justify='start'),
        rink.draw()
    ])
    return


if __name__ == "__main__":
    app.run()
