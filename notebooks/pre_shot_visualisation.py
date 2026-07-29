import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    from hockey_rink import NHLRink
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    from data_readers import batch_read_entity_tracking, read_game_id_mapping
    from pre_shot.features import PreShotData


@app.cell
def _():
    game_id_mapping = read_game_id_mapping("mappings/NHL_20252026_game_smt_sportlogiq_id_map.csv")
    player_id_mapping = pl.read_csv("mappings/NHL_20252026_player_sportlogiq_id_map.csv").cast(pl.String)

    game_ids = game_id_mapping.select(c('SportlogiqGameID')).to_series()
    game_id_selector = mo.ui.dropdown.from_series(game_ids, value=204631)
    game_id_selector
    return game_id_mapping, game_id_selector, player_id_mapping


@app.cell
def _(game_id_mapping, game_id_selector):
    games = (
        game_id_mapping
        .filter(
            c('SportlogiqGameID') == game_id_selector.value,
        )
    )

    sportlogiq_ids = games.select(c('SportlogiqGameID').cast(pl.String)).to_series().to_list()
    SMT_ids = games.select(c('SMTGameID')).to_series().to_list()

    shots = (
        pl.scan_parquet("output/post_shot_data_202510.parquet")
        .filter(c('game_id').is_in(sportlogiq_ids))
    )

    player_tracking = batch_read_entity_tracking(
        [f"data/smtoasis/*/games/{id}/*_entity_tracking_processed_measurements.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )
    return player_tracking, shots


@app.cell
def _(player_id_mapping, player_tracking, shots):
    pre_shot_data = PreShotData(shots, player_tracking, player_id_mapping)
    return (pre_shot_data,)


@app.cell
def _(pre_shot_data):
    shots_with_tracking = pre_shot_data.defender_data().collect()
    return (shots_with_tracking,)


@app.cell
def _(shots_with_tracking):
    shot_ids = shots_with_tracking.select(c('shot_id').unique()).to_series()

    shot_id_selector = mo.ui.dropdown.from_series(shot_ids, value=shot_ids.first())

    shot_id_selector
    return (shot_id_selector,)


@app.cell
def _(pre_shot_data, shot_id_selector, shots_with_tracking):
    shot_tracking = shots_with_tracking.filter(c('shot_id') == shot_id_selector.value)

    goalie_tracking = pre_shot_data.goalie_data().filter(c('shot_id') == shot_id_selector.value).collect()

    # shot_tracking.select(c('defender_id', 'dist_to_shooter', 'angle_to_shooter', 'inside_shooting_lane', 'inside_shadow_lane', 'pressure')).sort(c('pressure'), descending=True)
    return goalie_tracking, shot_tracking


@app.cell
def _(shot_tracking):
    lane_expansion = 3.0

    shot_x = shot_tracking.select(c('shot_x').first()).item()
    shot_y = shot_tracking.select(c('shot_y').first()).item()

    x_coords = [shot_x, 89, 89]
    y_coords = [shot_y, -3, 3]

    dx = 89 - shot_x
    dy = -shot_y
    dist = (dx ** 2 + dy ** 2) ** 0.5

    perp_x = -dy / dist
    perp_y = dx / dist

    sl_x = shot_x + (lane_expansion * perp_x)
    sl_y = shot_y + (lane_expansion * perp_y)

    sr_x = shot_x - (lane_expansion * perp_x)
    sr_y = shot_y - (lane_expansion * perp_y)

    shadow_x_coords = [sl_x, sr_x, 89, 89]
    shadow_y_coords = [sl_y, sr_y, -(3+lane_expansion), 3+lane_expansion]
    return shadow_x_coords, shadow_y_coords, x_coords, y_coords


@app.cell
def _(
    goalie_tracking,
    shadow_x_coords,
    shadow_y_coords,
    shot_tracking,
    x_coords,
    y_coords,
):
    rink = NHLRink()
    fig, ax = plt.subplots()
    rink.draw(display_range='offense', rotation=90)

    rink.arrow(
        x=shot_tracking.select(c('x_adj')), 
        y=shot_tracking.select(c('y_adj')),
        dx=shot_tracking.select(c('vx_adj')), 
        dy=shot_tracking.select(c('vy_adj')),
        facecolor='black',
        alpha=0.4,
        # draw_kw={'display_range': 'ozone', 'rotation': 90}
    )

    rink.scatter(
        x=shot_tracking.select(c('x_adj')),
        y=shot_tracking.select(c('y_adj')),
        s=100,
        c=shot_tracking.select(c('pressure')),
        cmap='Reds',
        vmin=0,
        vmax=1,
        edgecolors='black',
    )

    # Goalie Tracking
    rink.scatter(
        x=goalie_tracking.select(c('x_adj')),
        y=goalie_tracking.select(c('y_adj')),
        s=120,
        c='purple'
    )

    rink.arrow(
        x=goalie_tracking.select(c('x_adj')),
        y=goalie_tracking.select(c('y_adj')),
        dx=goalie_tracking.select(c('vx_adj')),
        dy=goalie_tracking.select(c('vy_adj')),
        facecolor='purple',
        alpha=0.7
    )


    # rink.text(
    #     x=shot_tracking.select(c('x_adj')) + 2,
    #     y=shot_tracking.select(c('y_adj')) - 1, 
    #     s=shot_tracking.select(c('pressure').round(2)),
    #     fontsize=8,
    # )

    rink.scatter(
        shot_tracking.select(c('shot_x')),
        shot_tracking.select(c('shot_y')),
        s=30,
        c='blue',
    )

    x_trans, y_trans = rink.convert_xy(x_coords, y_coords, ax=ax)
    lane = Polygon(
        list(zip(x_trans, y_trans)), 
        closed=True, 
        facecolor='cyan', 
        edgecolor='blue', 
        alpha=0.5,
        linewidth=2,
        zorder=10
    )

    shadow_x_trans, shadow_y_trans = rink.convert_xy(shadow_x_coords, shadow_y_coords, ax=ax)
    shadow_lane = Polygon(
        list(zip(shadow_x_trans, shadow_y_trans)), 
        closed=True,
        facecolor='deepskyblue',
        edgecolor='blue',
        alpha=0.3,
        linewidth=1,
        linestyle="--",
        zorder=9
    )

    ax.add_patch(lane)
    ax.add_patch(shadow_lane)
    return


if __name__ == "__main__":
    app.run()
