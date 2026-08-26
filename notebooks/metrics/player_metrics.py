import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    from great_tables import GT
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, theme_bw, labs
    from mizani.formatters import percent_format

    from data_readers import batch_read_shot_data
    from models.data import prepare_data

    return (
        GT,
        aes,
        batch_read_shot_data,
        c,
        cs,
        geom_point,
        ggplot,
        labs,
        p9,
        pl,
        prepare_data,
        theme_bw,
    )


@app.cell
def _(c, pl):
    reg_season_game_ids = (
        pl.scan_csv("mappings/NHL_20242025_20252026_game_smt_sportlogiq_id_map.csv")
        .filter(c('Stage') == 'regular')
        .select(c('SportlogiqGameID').cast(pl.String))
        .collect()
        .to_series()
    )
    return (reg_season_game_ids,)


@app.cell
def _(batch_read_shot_data, c, pl, reg_season_game_ids):
    shot_data_2425 = batch_read_shot_data("/output/shot_data/20242025/*.parquet").with_columns(season=pl.lit('20242025'))
    shot_data_2526 = batch_read_shot_data("/output/shot_data/20252026/*.parquet").with_columns(season=pl.lit('20252026'))

    shot_data = pl.concat([shot_data_2425, shot_data_2526], how='vertical').filter(c('game_id').is_in(reg_season_game_ids.implode()))
    return (shot_data,)


@app.cell
def _(c, pl):
    pre_shot_xg = pl.scan_parquet("/output/predictions/*/pre_shot_2008.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/*/post_shot_2108.parquet")

    player_mappings = pl.scan_csv("mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv").with_columns(c('SportlogiqPlayerID').cast(pl.String)).drop('EntityOfficialID')
    return player_mappings, post_shot_xg, pre_shot_xg


@app.cell
def _(
    c,
    pl,
    player_mappings,
    post_shot_xg,
    pre_shot_xg,
    prepare_data,
    shot_data,
):
    data = (
        shot_data
        .pipe(prepare_data)
        .join(
            pre_shot_xg,
            on=["game_id", "period", "shot_id"],
            how='left',
            validate='1:1'
        ).join(
            post_shot_xg,
            on=["game_id", "period", "shot_id"],
            how='left',
            validate='1:1'
        ).join(
            player_mappings,
            left_on="opposing_team_goalie_on_ice_ref",
            right_on="SportlogiqPlayerID",
            how='left',
            validate='m:1'
        ).rename({'PlayerName': "opposing_goaltender_name"})
        .with_columns(pl.when(c('type').str.contains('blocked')).then(0).otherwise(c('post_shot')).alias('post_shot'))
        .with_columns(c('post_shot').fill_null(c('pre_shot')))
        .collect()
    )
    return (data,)


@app.cell
def _(c, data, pl):
    shooter_metrics = (
        data
        .filter(c('season') == '20252026')
        .group_by("player_reference_id", "player_first_name", "player_last_name", "position")
        .agg(
            c('pre_shot').sum().alias('pre_shot_xg'),
            c('post_shot').sum().alias('post_shot_xg'),
            (c('goal') - c('pre_shot')).sum().alias('pre_gsax'),
            (c('goal') - c('post_shot')).sum().alias('post_gsax'),
            (c('post_shot') - c('pre_shot')).sum().alias('shooting_goals_added'),
            pl.len().alias('shots')
        ).with_columns(
            c('pre_gsax').rank(method='min', descending=True).alias('pre_gsax_rank')
        ).with_columns(
            pl.concat_str([c('player_first_name'), c('player_last_name')], separator=" ").alias('name')
        )
    )
    return (shooter_metrics,)


@app.cell
def _(c, data, pl):
    goalie_metrics = (
        data
        .filter(c('season') == '20252026')
        .group_by("opposing_team_goalie_on_ice_ref", "opposing_goaltender_name")
        .agg(
            c('pre_shot').sum().alias('pre_shot_xg'),
            c('post_shot').sum().alias('post_shot_xg'),
            (c('pre_shot') - c('goal')).sum().alias('pre_gsax'),
            (c('post_shot') - c('goal')).sum().alias('post_gsax'),
            pl.len().alias('shots_faced'),
            c('goal').sum().alias('goals_against')
        ).sort('post_gsax', descending=True)
        .with_row_index("rank", offset=1)
        .with_columns(
            diff = c('post_gsax') - c('pre_gsax')
        )
    )
    return (goalie_metrics,)


@app.cell
def _(c, shooter_metrics):
    forwards_table = (
        shooter_metrics
        .filter(c('position') != 'D')
        .select(c('name', 'shots', 'pre_shot_xg', 'post_shot_xg', 'shooting_goals_added'))
        .sort(c('shooting_goals_added'), descending=True)
        .with_row_index("rank", offset=1)
    )

    defensemen_table = (
        shooter_metrics
        .filter(c('position') == 'D')
        .select(c('name', 'shots', 'pre_shot_xg', 'post_shot_xg', 'shooting_goals_added'))
        .sort(c('shooting_goals_added'), descending=True)
        .with_row_index("rank", offset=1)
    )
    return defensemen_table, forwards_table


@app.cell
def _(GT, cs, forwards_table):
    top_10_shooters = (
        GT(
            forwards_table
            .head(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    top_10_shooters
    # print(top_10_shooters.as_latex())
    return


@app.cell
def _(GT, cs, forwards_table):
    bottom_10_shooters = (
        GT(
            forwards_table
            .tail(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    bottom_10_shooters
    # print(bottom_10_shooters.as_latex())
    return


@app.cell
def _(GT, cs, defensemen_table):
    top_5_defensemen = (
        GT(
            defensemen_table
            .head(5)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    top_5_defensemen
    # print(top_5_defensemen.as_latex())
    return


@app.cell
def _(GT, cs, defensemen_table):
    bottom_5_defensemen = (
        GT(
            defensemen_table
            .tail(5)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    bottom_5_defensemen
    # print(bottom_5_defensemen.as_latex())
    return


@app.cell
def _(GT, cs, goalie_metrics):
    top_10_goalies = (
        GT(
            goalie_metrics
            .select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax', 'diff')
            .head(10)
        ).cols_label(rank="Rank", opposing_goaltender_name="Goaltender", pre_gsax="Pre-Shot GSAX", post_gsax="Post-Shot GSAX", diff="Difference")
        .fmt_number(cs.numeric().exclude('rank'))
        .fmt_number('diff', force_sign=True)
        .data_color('diff', domain=(-15, 15), palette=["red", "white", "green"])
    )

    top_10_goalies
    # print(top_10_goalies.as_latex())
    return


@app.cell
def _(GT, cs, goalie_metrics):
    bottom_10_goalies = (
        GT(
            goalie_metrics
            .select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax', 'diff')
            .tail(10)
        ).cols_label(rank="Rank", opposing_goaltender_name="Goaltender", pre_gsax="Pre-Shot GSAX", post_gsax="Post-Shot GSAX", diff="Difference")
        .fmt_number(cs.numeric().exclude('rank'))
        .fmt_number('diff', force_sign=True)
        .data_color('diff', domain=(-15, 15), palette=["red", "white", "green"])
    )

    bottom_10_goalies
    # print(bottom_10_goalies.as_latex())
    return


@app.cell
def _(c, data):
    player_data = (
        data
        .filter(
            c('season') == '20252026',
            c('player_first_name') == 'Connor',
            c('player_last_name') == 'Bedard',
            c('type').str.contains('blocked').not_(),
            c('goalline_y').is_not_null(),
            c('goalline_z').is_not_null()
        ).with_columns(
            sga=c('post_shot') - c('pre_shot')
        )
        # .select(
        #     c('pre_shot').sum().alias('pre_shot_xg'),
        #     c('post_shot').sum().alias('post_shot_xg'),
        #     (c('goal') - c('pre_shot')).sum().alias('pre_gsax'),
        #     (c('goal') - c('post_shot')).sum().alias('post_gsax'),
        #     (c('post_shot') - c('pre_shot')).sum().alias('shooting_goals_added'),
        #     pl.len().alias('shots')
        # )
    )

    sga = player_data.select(c('sga').sum()).item()
    return player_data, sga


@app.cell
def _(aes, geom_point, ggplot, labs, p9, pl, player_data, sga, theme_bw):
    net_outline = pl.DataFrame({
        'y': [-3, -3, 3, 3],
        'z': [0, 4, 4, 0]
    })

    player_plot = (
        ggplot()
        + p9.coord_fixed(ratio=1, ylim=(0, 6), xlim=(5, -5))
        # Net
        + p9.geom_rect(
            aes(xmin=-3, xmax=3, ymin=0, ymax=4), fill='lightgrey',
        )
        # Ice Surface
        + p9.geom_segment(
            aes(x=-10, xend=10, y=0, yend=0), 
            color="lightblue", size=2
        ) 
        # Posts
        + p9.geom_path(
            aes(x='y', y='z'), 
            data=net_outline, 
            color="red", size=2, lineend="round"
        ) + geom_point(
            aes(x='goalline_y', y='goalline_z', size='pre_shot', fill='sga'), data=player_data
        ) + p9.geom_label(aes(x=4, y=5.5), data=None, label=f"SGA: {sga:.2f}")
        + p9.scale_fill_gradient2(low="Red", mid="White", high="Green")
        + p9.scale_x_reverse()
        + theme_bw(base_size=12)
        + labs(x="", y="", fill="Shooting Goals Added", size="Pre-Shot xG",
               caption="2025-26 Reg. Season Unblocked Shots",)
        + p9.theme(legend_title=p9.element_text(angle=-90), legend_title_position="right", legend_key_width=12,
                   axis_text=p9.element_blank(), axis_ticks=p9.element_blank())
    )

    # player_plot.save("plots/metrics/bedard_shot_chart.png", dpi=500)
    player_plot
    return (net_outline,)


@app.cell
def _(c, data):
    goalie_data = (
        data
        .filter(
            c('season') == '20252026',
            c('opposing_goaltender_name') == 'Scott Wedgewood',
            c('type').str.contains('blocked').not_(),
            c('goalline_y').is_not_null(),
            c('goalline_z').is_not_null(),
        )
        # .select(
        #     c('pre_shot').sum().alias('pre_shot_xg'),
        #     c('post_shot').sum().alias('post_shot_xg'),
        #     (c('goal') - c('pre_shot')).sum().alias('pre_gsax'),
        #     (c('goal') - c('post_shot')).sum().alias('post_gsax'),
        #     (c('post_shot') - c('pre_shot')).sum().alias('shooting_goals_added'),
        #     pl.len().alias('shots')
        # )
    )
    return (goalie_data,)


@app.cell
def _(aes, geom_point, ggplot, goalie_data, labs, net_outline, p9, theme_bw):
    goalie_plot = (
        ggplot()
        + p9.coord_fixed(ratio=1, ylim=(0, 5), xlim=(4, -4))
        # Net
        + p9.geom_rect(
            aes(xmin=-3, xmax=3, ymin=0, ymax=4), fill='lightgrey',
        )
        # Ice Surface
        + p9.geom_segment(
            aes(x=-10, xend=10, y=0, yend=0), 
            color="lightblue", size=2
        ) 
        # Posts
        + p9.geom_path(
            aes(x='y', y='z'), 
            data=net_outline, 
            color="red", size=2, lineend="round"
        ) + geom_point(
            aes(x='goalline_y_norm', y='goalline_z', size='post_shot', fill='goal', alpha='goal'),
            data=goalie_data,
        )
        + p9.scale_fill_discrete(limits=(0,1), labels=("Save", "Goal"), direction=-1)
        + p9.scale_alpha_discrete(range=(0.2,1), limits=(0,1), labels=("Save", "Goal"), guide=None)
        + p9.scale_x_reverse()
        + theme_bw(base_size=12)
        + labs(x="", y="", fill="Outcome", size="Post-Shot xG",
               subtitle="← Blocker Side       Glove Side →",
               caption="2025-26 Reg. Season Unblocked Shots",)
        + p9.theme(
            legend_key_width=12, plot_subtitle=p9.element_text(ha="center"),
            axis_text=p9.element_blank(), axis_ticks=p9.element_blank()
        ) + p9.guides(fill=p9.guide_legend(override_aes={'size':8}))
    )

    # goalie_plot.save("plots/metrics/markstrom_shot_chart.png", dpi=500)
    goalie_plot
    return


@app.cell
def _(c, data, pl):
    season_pairs = (
        data
        .with_columns(pl.when(c('position') == 'D').then(pl.lit('Defenders')).otherwise(pl.lit('Forwards')).alias('position_group'))
        .group_by("season", "player_reference_id", "position_group")
        .agg(
            c('pre_shot').sum().alias('pre_shot_xg'),
            c('post_shot').sum().alias('post_shot_xg'),
            c('goal').sum().alias('goals_scored'),
            (c('post_shot') - c('pre_shot')).sum().alias('shooting_goals_added'),
            pl.len().alias('shots')
        ).filter(c('shots') >= 150)
        .pivot(
            on="season",
            values="shooting_goals_added",
            index=["player_reference_id", "position_group"]
        ).drop_nulls()
    )
    return (season_pairs,)


@app.cell
def _(aes, geom_point, ggplot, labs, p9, pl, season_pairs, theme_bw):
    r_squared = season_pairs.select(pl.corr("20242025", "20252026").pow(2)).item()

    corr_plot = (
        ggplot(season_pairs, aes(x="20242025", y="20252026", colour="position_group"))
        + geom_point()
        + p9.geom_smooth(method="lm", se=False, colour="black")
        + p9.geom_label(label=f"R^2={r_squared:.2f}", x=-2.5, y=5, colour="black")
        + theme_bw(base_size=12)
        + labs(x="2024-2025 SGA", y="2025-2026 SGA", colour="Position", caption="min. 150 Shot Attempts")
        # + p9.coord_fixed(xlim=(0.08,0.52), ylim=(0.08,0.52))
        # + p9.scale_x_continuous(labels=percent_format())
        # + p9.scale_y_continuous(labels=percent_format())
    )

    # corr_plot.save("plots/metrics/sga_correlations.svg")
    corr_plot
    return


if __name__ == "__main__":
    app.run()
