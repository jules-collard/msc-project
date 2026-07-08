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

    from tracking_processing import derive_game_clock
    from data_readers import read_entity_registration, read_entity_tracking, read_events, read_id_mapping
    from event_processing import add_pass_target, add_carry_info, join_tracking


@app.cell
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json")
    shots = events.filter(c('name') == 'shot')

    mapping = read_id_mapping("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_player_sportlogiq_id_map.csv")

    tracking = (
        read_entity_tracking(
        "data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet"
        ).pipe(derive_game_clock)
    )

    rosters = read_entity_registration("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_registration.json", lazy=False)
    return mapping, rosters, shots, tracking


@app.cell
def _(mapping, rosters, shots, tracking):
    shot_tracking = (
        join_tracking(shots, tracking, mapping)
        .join(
            rosters.lazy().select(c('OfficialId', 'JerseyNum')),
            left_on='onice_player_ref',
            right_on='OfficialId',
            how='left'
        ).collect()
    )
    return (shot_tracking,)


@app.cell
def _(shot_tracking):
    times = shot_tracking.select(c('game_time').unique()).to_series()
    game_time_selector = mo.ui.dropdown.from_series(
        times,
        allow_select_none=False,
        value=times.first()
    )
    return (game_time_selector,)


@app.cell
def _(display_data):
    display_data
    return


@app.cell
def _(game_time_selector, shot_tracking):
    display_data = shot_tracking.filter(c('game_time') == game_time_selector.value)
    current_event = display_data.select(pl.all().first()).to_dicts()[0]

    rink = NHLRink()

    # rink.arrow(
    #     x=display_data.select(c('x')), 
    #     y=display_data.select(c('y')),
    #     dx=display_data.select(c('vx')), 
    #     dy=display_data.select(c('vy'))
    # )

    rink.scatter(
        x=display_data.select(c('x_adj')), 
        y=display_data.select(c('y_adj')),
        c=display_data.select(c('team_source').replace({'with_team': 'tab:red', 'opposing_team': 'tab:blue'})).to_series(),
        s=250, 
        edgecolor='black'
    )

    rink.scatter(
        x=display_data.select(c('x_adj').first()),
        y=display_data.select(c('y_adj').first()),
        c='black',
        s=120
    )

    rink.text(
        display_data.select(c('x_adj')), 
        display_data.select(c('y_adj')), 
        display_data.select(c('JerseyNum')), 
        fontsize=10, 
        ha="center", 
        va="center", 
        color="white"
    )

    plt.suptitle(f"{current_event['player_last_name']} {current_event['player_jersey']} - {current_event['shorthand']}: {current_event['outcome']}")
    plt.title(f"{current_event['flags']}")

    mo.vstack([
        game_time_selector,
        rink.draw()
    ])
    return (display_data,)


if __name__ == "__main__":
    app.run()
