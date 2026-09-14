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
    import mizani.labels as ml

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
        ml,
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

    pre_shot_xg = pl.scan_parquet("/output/predictions/*/pre_shot_0809.parquet").filter(c('game_id') == str(sportlogiq_id)).collect()
    post_shot_xg = pl.scan_parquet("/output/predictions/*/post_shot_one_hot_0809.parquet").filter(c('game_id') == str(sportlogiq_id)).collect()

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
        .join(
            pre_shot_xg,
            on=['game_id', 'period', 'shot_id'],
            how='left'
        ).join(
            post_shot_xg,
            on=['game_id', 'period', 'shot_id'],
            how='left'
        )
        .with_columns(
            c('shot_x').fill_null(c('x_adj_coord')),
            c('shot_y').fill_null(c('y_adj_coord')),
            c('shot_time').fill_null(c('elapsed_time')),
            result = (
                pl.when(c('goal')).then(pl.lit('Goal'))
                .when(c('type').str.contains('blocked')).then(pl.lit('Blocked'))
                .when(c('outcome') == 'successful').then(pl.lit('On Goal'))
                .otherwise(pl.lit('Missed'))
            )
        ).with_columns(
            details = pl.format("{} {} - {}mph - {}", c('player_first_name'), c('player_last_name'), c('shot_speed').round(1), c('result'))
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
    return puck_tracking, shot_data


@app.cell
def _(c, cs, pl, sportlogiq_id):
    shap = (
        pl.read_parquet(f"/output/shap/*/{sportlogiq_id}_shap.parquet")
        .unpivot(
            on=cs.all().exclude('game_id', 'period', 'shot_id', 'base_value'),
            index=['game_id', 'period', 'shot_id', 'base_value'],
            variable_name='variable',
            value_name='shap'
        ).with_columns(c('shap').abs().alias('abs_shap'))
        .sort('shot_id', 'abs_shap')
        .with_columns(
            (c('base_value') + c('shap').cum_sum().over('shot_id')).alias('end'),
            (c('shap') > 0).alias('increase')
        ).with_columns(
            c('end').shift(1).over('shot_id').fill_null(c('base_value')).alias('start')
        ).with_columns(
            c('variable').str.replace("_", " ").str.to_titlecase()
        )
    )
    return (shap,)


@app.cell
def _(c, mo, shot_data):
    max_index = shot_data.select(c('index').max()).item()
    shot_selector = mo.ui.number(start=1, stop=max_index, value=1, step=1)
    return (shot_selector,)


@app.cell
def _(c, pl, puck_tracking, shap, shot_data, shot_selector, time):
    shot_tracking = shot_data.filter(c('index') == shot_selector.value)
    shot_plot_data = shot_tracking.head(1)

    shot_id = shot_plot_data.select(c('shot_id').first()).item()
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

    shot_shap = (
        shap
        .filter(c('shot_id') == shot_id)
        .with_row_index('x_id', offset=1)
    )

    base_value = shot_shap.select(c('base_value').first()).item()
    details = shot_plot_data.select(c('details').first()).item()
    return (
        base_value,
        details,
        period,
        shot_plot_data,
        shot_puck_tracking,
        shot_shap,
        shot_tracking,
        time_remaining,
    )


@app.cell
def _(c, shot_plot_data):
    pre_xg = shot_plot_data.select(c('pre_shot').round(2).first()).item()
    post_xg = shot_plot_data.select(c('post_shot').round(2).first()).item()
    if post_xg is None:
        post_colour = 'grey'
    elif post_xg > pre_xg:
        post_colour = 'green'
    else:
        post_colour = 'red'
    return post_colour, post_xg, pre_xg


@app.cell
def _(
    aes,
    details,
    game_selector,
    geom_net,
    gg,
    ggplot,
    period,
    post_colour,
    post_xg,
    pre_xg,
    shot_plot_data,
    time_remaining,
):
    goal_plot = (
        ggplot(shot_plot_data, aes(x='goalline_y', y='goalline_z'))
        + geom_net()
        # + geom_ice()
        + gg.geom_point(size=5)
        + gg.geom_label(
            aes(x=-5, y=5, alpha=0.8), label=f"PreXG: {pre_xg}", size=15, show_legend=False
        ) + gg.geom_label(
            aes(x=-5, y=4, alpha=0.8), label=f"PostXG: {post_xg}", colour=post_colour, size=15, show_legend=False
        )
        + gg.scale_x_reverse()
        + gg.coord_fixed(xlim=(6,-6), ylim=(0, 6))
        + gg.theme_538()
        + gg.theme(
            axis_text=gg.element_blank(), axis_ticks=gg.element_blank(),
            axis_title=gg.element_blank(),
            plot_title=gg.element_text(size=20, weight="bold"),
            dpi=300
        )
        + gg.labs(
            title=game_selector.value,
            subtitle=f"Period {period}, {time_remaining} remaining \n {details}"
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
def _(aes, base_value, c, gg, ggplot, ml, shot_shap):
    if shot_shap.height > 0:
        max_shap = shot_shap.select(c('end').max()).item()

        shap_plot = (
            ggplot(shot_shap)
            + gg.geom_segment(
                aes(
                    x="x_id - 1.4", 
                    xend="x_id + 0.4", 
                    y="start", 
                    yend="start"
                ),
                linetype="dashed", 
                color="gray",
                inherit_aes=False
            )
            + gg.geom_rect(
                aes(
                    xmin="x_id - 0.4", 
                    xmax="x_id + 0.4", 
                    ymin="start", 
                    ymax="end", 
                    fill="increase"
                ),
                size=0.5
            )
            + gg.geom_hline(yintercept=base_value, linetype="dotted")
            + gg.geom_label(
                aes(
                    x="x_id",
                    y=max_shap + 0.01,
                    label="shap",
                    colour="increase"
                ),
                format_string="{:+.1%}",
                size=8
            )
            + gg.scale_x_continuous(
                breaks=shot_shap["x_id"].to_list(),
                labels=shot_shap["variable"].to_list(),
                limits=(0.25, None)
            ) + gg.scale_y_continuous(
                limits=(None, max_shap + 0.02),
                labels=ml.label_percent()
            )
            + gg.scale_fill_manual(values={True: "green", False: "red"}) # Standard SHAP colors
            + gg.scale_color_manual(values={True: "green", False: "red"}) # Standard SHAP colors
            + gg.theme_538(base_size=8)
            + gg.coord_flip()
            + gg.theme(
                panel_grid_major_y=gg.element_blank(),
                panel_border=gg.element_rect(size=0.5),
                legend_position="none",
                dpi=300,
                axis_text_y=gg.element_text(weight='bold')
            )
            + gg.labs(
                x="", y="Post-Shot xG"
            )
        )
    else:
        shap_plot = None
    return (shap_plot,)


@app.cell
def _(game_selector, goal_plot, mo, rink_plot, shap_plot, shot_selector):
    mo.vstack([
        mo.hstack([
            game_selector, shot_selector
        ], justify="start"),
        mo.hstack([goal_plot, rink_plot], widths=[1,1], gap=0),
        mo.hstack([shap_plot])
    ], gap=0)
    return


if __name__ == "__main__":
    app.run()
