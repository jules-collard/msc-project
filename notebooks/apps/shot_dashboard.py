import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from datetime import date
    import time

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from polars import col as c
    import plotnine as gg
    from plotnine import ggplot, aes

    from data_readers import read_game_id_mapping, batch_read_shot_data, batch_read_entity_tracking, batch_read_puck_tracking, read_player_id_mapping
    from plotting.rink import geom_ice, geom_net, geom_rink
    from processing.tracking import calculate_elapsed_time, adjust_vectors

    return (
        adjust_vectors,
        aes,
        batch_read_entity_tracking,
        batch_read_puck_tracking,
        batch_read_shot_data,
        c,
        calculate_elapsed_time,
        cs,
        date,
        geom_net,
        geom_rink,
        gg,
        ggplot,
        mo,
        pl,
        read_game_id_mapping,
        read_player_id_mapping,
        time,
    )


@app.cell
def _(c, date, pl, read_game_id_mapping, read_player_id_mapping):
    games = (
        read_game_id_mapping("mappings/NHL_20242025_20252026_game_smt_sportlogiq_id_map.csv")
        .with_columns(
            pl.format("{}@{} {}", c('AwayTeam'), c('HomeTeam'), c('GameDate')).alias('game_string')
        ).filter(c('GameDate') >= date(2025, 10, 1)) # 25-26 season
    )

    player_mapping = read_player_id_mapping("mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv")
    return games, player_mapping


@app.cell
def _(c, games, mo):
    game_strings = games.select(c('game_string')).to_series()
    game_selector = mo.ui.dropdown(options=game_strings, searchable=True, label="Game: ", allow_select_none=False, value=game_strings.first())
    return (game_selector,)


@app.cell
def _(c, game_selector, games):
    sportlogiq_id = games.filter(c('game_string') == game_selector.value).select(c('SportlogiqGameID')).item()
    smt_id = games.filter(c('game_string') == game_selector.value).select(c('SMTGameID')).item()
    return smt_id, sportlogiq_id


@app.cell
def _(
    adjust_vectors,
    batch_read_entity_tracking,
    batch_read_puck_tracking,
    batch_read_shot_data,
    c,
    calculate_elapsed_time,
    cs,
    games,
    pl,
    player_mapping,
    smt_id,
    sportlogiq_id,
):
    shot_data = batch_read_shot_data(f"/output/shot_data/*-wide-window/{sportlogiq_id}_shot_data.parquet").with_row_index(offset=1).collect()

    tracking = (
        batch_read_entity_tracking(f"/data/smtoasis/regular/games/{smt_id}/*_entity_tracking_processed_measurements.parquet", games.lazy())
        .join(
            player_mapping.lazy(),
            left_on='entity_official_id',
            right_on='EntityOfficialID',
            how='left'
        ).with_columns(elapsed_time = calculate_elapsed_time())
        .drop(c('segment_idx'), cs.starts_with('kappa'))
        .collect()
    )

    puck_tracking = (
        batch_read_puck_tracking(
            f"/data/smtoasis/regular/games/{smt_id}/*_puck_tracking_raw_measurements.parquet",
            mapping=games.lazy()
        ).with_columns(elapsed_time = calculate_elapsed_time())
        .collect()
    )

    shot_data = (
        shot_data
        .with_columns(
            c('shot_x').fill_null(c('x_adj_coord')),
            c('shot_y').fill_null(c('y_adj_coord')),
            c('shot_time').fill_null(c('elapsed_time'))
        )
        .with_columns(
            pl.concat_list(c('opposing_team_goalie_on_ice_ref')),
            with_team = pl.concat_list(c('team_forwards_on_ice_refs', 'team_defencemen_on_ice_refs')).list.drop_nulls(),
            opposing_team = pl.concat_list(c('opposing_team_forwards_on_ice_refs', 'opposing_team_defencemen_on_ice_refs')).list.drop_nulls(),
        )
        .unpivot(
            on=['with_team', 'opposing_team', 'opposing_team_goalie_on_ice_ref'], 
            variable_name='team_source', 
            value_name='onice_player_ref', 
            index=cs.all() - cs.by_name('with_team', 'opposing_team', 'opposing_team_goalie_on_ice_ref'),
        ).explode('onice_player_ref', empty_as_null=True)
        .drop(cs.ends_with('on_ice_refs'), cs.ends_with('on_ice_ref'))
        .sort('period', 'onice_player_ref', 'shot_time')
        .join_asof(
            tracking.sort('period', 'SportlogiqPlayerID', 'elapsed_time'), 
            left_on='shot_time',
            right_on='elapsed_time',
            by_left=['period', 'onice_player_ref'], 
            by_right=['period', 'SportlogiqPlayerID'],
            tolerance=0.15,
            strategy='nearest',
            check_sortedness=False
        ).with_columns(
            adjust_vectors(c('x', 'y', 'vx', 'vy', 'ax', 'ay')),
            c('team_source').replace_strict({'with_team': 'Offense', 'opposing_team': 'Defence', 'opposing_team_goalie_on_ice_ref': 'Goalie'})
        )
    )

    shap = pl.read_parquet(f"/output/shap/*/{sportlogiq_id}_shap.parquet")
    return puck_tracking, shot_data


@app.cell
def _(c, mo, shot_data):
    max_index = shot_data.select(c('index').max()).item()
    shot_selector = mo.ui.number(start=1, stop=max_index, value=1, step=1)
    return (shot_selector,)


@app.cell
def _(c, pl, puck_tracking, shot_data, shot_selector, time):
    shot_tracking = shot_data.filter(c('index') == shot_selector.value)
    shot_plot_data = shot_tracking.head(1)

    period = shot_plot_data.select(c('period').first()).item()
    elapsed_time = shot_plot_data.select(c('elapsed_time').first()).item()
    flip = shot_plot_data.select(c('flip').first()).item()
    time_remaining = time.strftime("%M:%S", time.gmtime(period * 1200 - shot_plot_data.select(c('game_time').first()).item()))

    shot_puck_tracking = (
        puck_tracking
        .filter(
            c('period') == period,
            (c('elapsed_time') - elapsed_time).abs() <= 0.75
        )
        .with_columns(
            pl.when(flip).then(-c('x','y','z')).otherwise(c('x','y','z'))
        )
    )
    return (
        period,
        shot_plot_data,
        shot_puck_tracking,
        shot_tracking,
        time_remaining,
    )


@app.cell
def _(
    aes,
    game_selector,
    geom_net,
    gg,
    ggplot,
    period,
    shot_plot_data,
    time_remaining,
):
    goal_plot = (
        ggplot(shot_plot_data, aes(x='goalline_y', y='goalline_z'))
        + geom_net()
        # + geom_ice()
        + gg.geom_point(size=5)
        + gg.scale_x_reverse()
        + gg.coord_fixed(xlim=(6,-6), ylim=(0, 6))
        + gg.theme_538()
        + gg.theme(
            axis_text=gg.element_blank(), axis_ticks=gg.element_blank(),
            axis_title=gg.element_blank(),
            dpi=300
        ) + gg.labs(
            title=game_selector.value,
            subtitle=f"Period {period}, {time_remaining} remaining"
        )
    )
    return (goal_plot,)


@app.cell
def _(
    aes,
    geom_rink,
    gg,
    ggplot,
    shot_plot_data,
    shot_puck_tracking,
    shot_tracking,
):
    rink_plot = (
        ggplot(shot_plot_data, aes(x='shot_x', y='shot_y'))
        + geom_rink(fill='white')
        + gg.geom_point(
            aes(x='x', y='y'), fill='grey', alpha=0.2, data=shot_puck_tracking
        )
        + gg.geom_point(
            aes(x='x_adj', y='y_adj', fill='team_source'),
            size=8,
            data=shot_tracking,
            inherit_aes=False)
        + gg.geom_point()
        + gg.geom_segment(aes(xend=89, yend='goalline_y'), size=0.8, arrow=gg.arrow(length=0.1, type='closed'))
        + gg.coord_fixed(xlim=(25,None))
        + gg.theme_538()
        + gg.theme(
            panel_grid=gg.element_blank(),
            axis_title=gg.element_blank(),
            axis_text=gg.element_blank(),
            legend_title=gg.element_blank()
        )
    )
    return (rink_plot,)


@app.cell
def _(game_selector, goal_plot, mo, rink_plot, shot_selector):
    mo.vstack([
        mo.hstack([
            game_selector, shot_selector
        ], justify="start"),
        mo.hstack([goal_plot, rink_plot], widths=[1,1], gap=0)
    ])
    return


if __name__ == "__main__":
    app.run()
