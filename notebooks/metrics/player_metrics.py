import marimo

__generated_with = "0.24.2"
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
    shot_data_2425 = batch_read_shot_data("/output/shot_data/20242025-wide-window/*.parquet").with_columns(season=pl.lit('20242025'))
    shot_data_2526 = batch_read_shot_data("/output/shot_data/20252026-wide-window/*.parquet").with_columns(season=pl.lit('20252026'))

    shot_data = pl.concat([shot_data_2425, shot_data_2526], how='vertical').filter(c('game_id').is_in(reg_season_game_ids.implode()))
    return (shot_data,)


@app.cell
def _(c, pl):
    pre_shot_xg = pl.scan_parquet("/output/predictions/*/pre_shot_0809.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/*/post_shot_one_hot_0809.parquet")

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
        .data_color(
            columns=['pre_shot_xg', 'post_shot_xg'],
            palette='Purples',
            domain=[0, 50]
        ).data_color(
            columns=['shooting_goals_added'],
            palette=['red', 'white', 'green'],
            domain=[-10, 10]
        )
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
        .data_color(
            columns=['pre_shot_xg', 'post_shot_xg'],
            palette='Purples',
            domain=[0, 50]
        ).data_color(
            columns=['shooting_goals_added'],
            palette=['red', 'white', 'green'],
            domain=[-10, 10]
        )
    )

    bottom_10_shooters
    # print(bottom_10_shooters.as_latex())
    return


@app.cell
def _(GT, cs, defensemen_table):
    top_5_defensemen = (
        GT(
            defensemen_table
            .head(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
        .data_color(
            columns=['pre_shot_xg', 'post_shot_xg'],
            palette='Purples',
            domain=[0, 50]
        ).data_color(
            columns=['shooting_goals_added'],
            palette=['red', 'white', 'green'],
            domain=[-10, 10]
        )
    )

    top_5_defensemen
    # print(top_5_defensemen.as_latex())
    return


@app.cell
def _(GT, cs, defensemen_table):
    bottom_5_defensemen = (
        GT(
            defensemen_table
            .tail(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
        .data_color(
            columns=['pre_shot_xg', 'post_shot_xg'],
            palette='Purples',
            domain=[0, 50]
        ).data_color(
            columns=['shooting_goals_added'],
            palette=['red', 'white', 'green'],
            domain=[-10, 10]
        )
    )

    bottom_5_defensemen
    # print(bottom_5_defensemen.as_latex())
    return


@app.cell
def _(GT, cs, goalie_metrics):
    top_10_goalies = (
        GT(
            goalie_metrics
            .select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax')
            .head(10)
        ).cols_label(rank="Rank", opposing_goaltender_name="Goaltender", pre_gsax="Pre-Shot GSAX", post_gsax="Post-Shot GSAX")
        .fmt_number(cs.numeric().exclude('rank'))
        .fmt_number(['pre_gsax', 'post_gsax'], force_sign=True)
        .data_color(
            columns=['pre_gsax', 'post_gsax'],
            palette=['red', 'white', 'green'],
            domain=[-40, 40]
        )
    )

    top_10_goalies
    # print(top_10_goalies.as_latex())
    return


@app.cell
def _(GT, cs, goalie_metrics):
    bottom_10_goalies = (
        GT(
            goalie_metrics
            .select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax')
            .tail(10)
        ).cols_label(rank="Rank", opposing_goaltender_name="Goaltender", pre_gsax="Pre-Shot GSAX", post_gsax="Post-Shot GSAX")
        .fmt_number(cs.numeric().exclude('rank'))
        .fmt_number(['pre_gsax', 'post_gsax'], force_sign=True)
        .data_color(
            columns=['pre_gsax', 'post_gsax'],
            palette=['red', 'white', 'green'],
            domain=[-40, 40]
        )
    )

    bottom_10_goalies
    # print(bottom_10_goalies.as_latex())
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
