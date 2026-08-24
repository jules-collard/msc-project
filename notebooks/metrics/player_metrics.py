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

    from data_readers import batch_read_shot_data
    from models.data import prepare_data

    return GT, batch_read_shot_data, c, cs, pl, prepare_data


@app.cell
def _(batch_read_shot_data, c, pl):
    shot_data = batch_read_shot_data("/output/shot_data/20252026-clean/*.parquet")
    pre_shot_xg = pl.scan_parquet("/output/predictions/20252026/post_shot_2108.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/20252026/pre_shot_2008.parquet")

    player_mappings = pl.scan_csv("mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv").with_columns(c('SportlogiqPlayerID').cast(pl.String)).drop('EntityOfficialID')
    return player_mappings, post_shot_xg, pre_shot_xg, shot_data


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
        .group_by("player_reference_id", "player_first_name", "player_last_name")
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
    sga_table = (
        shooter_metrics
        .select(c('name', 'shots', 'pre_shot_xg', 'post_shot_xg', 'shooting_goals_added'))
        .sort(c('shooting_goals_added'), descending=True)
        .with_row_index("rank", offset=1)
    )
    return (sga_table,)


@app.cell
def _(GT, cs, sga_table):
    top_10_shooters = (
        GT(
            sga_table
            .head(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    top_10_shooters
    # print(top_10_shooters.as_latex())
    return


@app.cell
def _(GT, c, cs, sga_table):
    bottom_10_shooters = (
        GT(
            sga_table
            .sort(c('rank'), descending=True)
            .head(10)
        ).cols_label(rank="Rank", name="Player Name", shots="Shot Attempts", pre_shot_xg="PreXG", post_shot_xg="PostXG", shooting_goals_added="SGA")
        .fmt_number(cs.numeric().exclude('rank', 'shots'))
    )

    bottom_10_shooters
    # print(bottom_10_shooters.as_latex())
    return


@app.cell
def _(GT, cs, goalie_metrics):
    top_10_goalies = (
        GT(
            goalie_metrics.select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax', 'diff')
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
            .sort('post_gsax')
            .select('rank', 'opposing_goaltender_name', 'pre_gsax', 'post_gsax', 'diff')
            .head(10)
        ).cols_label(rank="Rank", opposing_goaltender_name="Goaltender", pre_gsax="Pre-Shot GSAX", post_gsax="Post-Shot GSAX", diff="Difference")
        .fmt_number(cs.numeric().exclude('rank'))
        .fmt_number('diff', force_sign=True)
        .data_color('diff', domain=(-15, 15), palette=["red", "white", "green"])
    )

    bottom_10_goalies
    print(bottom_10_goalies.as_latex())
    return


@app.cell
def _(c, data):
    (
        data
        .group_by("opposing_team_goalie_on_ice_ref")
        .agg(
            c('post_shot').sum().alias('post_shot_xg'),
            c('goal').sum().alias('goals_against')
        ).filter(c('post_shot_xg') >= 30)
        .with_columns(
            multiplier = c('post_shot_xg') / c('goals_against')
        ).select(
            c('multiplier').mean().alias('mean'),
            c('multiplier').var().alias('variance')
        ).with_columns(
            beta = c('mean') / c('variance'),
            alpha = c('mean').pow(2) / c('variance')
        )
    )
    return


@app.cell
def _():
    alpha_0 = 110
    beta_0 = 110
    return alpha_0, beta_0


@app.cell
def _(alpha_0, beta_0, c, goalie_metrics):
    (
        goalie_metrics
        .with_columns(
            alpha_post = alpha_0 + c('goals_against'),
            beta_post = beta_0 + c('post_shot_xg')
        ).with_columns(
            skill_rating = c('beta_post') / c('alpha_post')
        )
    )
    return


if __name__ == "__main__":
    app.run()
