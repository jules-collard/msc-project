import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import json

    import marimo as mo
    import polars as pl
    from polars import col as c
    from hockey_rink import NHLRink

    from parquet_helpers import EntityTrackingReader


@app.cell
def _():
    with open("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet", mode="rb") as f:
        reader = EntityTrackingReader(f.read())

    tracking_pa = reader.get_table()
    tracking = pl.from_arrow(tracking_pa)
    return (tracking,)


@app.cell
def _():
    with open("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_registration.json") as roster_f:
        rosters_json = json.load(roster_f)

    rosters = pl.from_dicts(rosters_json[0].get("Entities"))
    return (rosters,)


@app.cell
def _(rosters, selector, tracking):
    player_ids = tracking.select(c('entity_official_id')).unique()
    timestamps = pl.from_dict({'ts': [selector.value]}, schema={'ts': pl.Float64()})
    ts_tracking = ( 
        timestamps
        .join(player_ids, how='cross').sort(c('ts'))
        .join(rosters, left_on='entity_official_id', right_on='OfficialId', how='left')
        .join_asof(
            tracking.sort(c('ts')),
            on=c('ts'), 
            by='entity_official_id', 
            strategy='forward', 
            tolerance=0.1
        ).drop_nulls(c('entity_id'))
    )
    return (ts_tracking,)


@app.cell
def _(ts_tracking):
    ts_tracking
    return


@app.cell
def _(tracking):
    selector = mo.ui.number(value=tracking.select(c('ts')).min().item(), step=0.1)
    return (selector,)


@app.cell
def _(selector, ts_tracking):
    rink = NHLRink()

    rink.arrow(
        x=ts_tracking.select(c('x')), 
        y=ts_tracking.select(c('y')),
        dx=ts_tracking.select(c('vx')), 
        dy=ts_tracking.select(c('vy'))
    )

    rink.scatter(
        x=ts_tracking.select(c('x')), 
        y=ts_tracking.select(c('y')),
        c=ts_tracking.select(c('VisOrHome').replace({'Home': 'tab:red', 'Visitor': 'tab:blue'})).to_series(),
        s=250, 
        edgecolor='black'
    )

    rink.text(
        ts_tracking.select(c('x')), 
        ts_tracking.select(c('y')), 
        ts_tracking.select(c('JerseyNum')), 
        fontsize=10, 
        ha="center", 
        va="center", 
        color="white"
    )

    mo.vstack([
        selector,
        rink.draw()
    ])
    return


if __name__ == "__main__":
    app.run()
