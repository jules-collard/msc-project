import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, theme_bw, labs
    from mizani.labels import label_percent, label_number
    from ninejs import interactive, css, to_html

    from data_readers import batch_read_shot_data
    from models.data import prepare_data
    from plotting.rink import geom_rink, geom_net

    return (
        aes,
        batch_read_shot_data,
        c,
        css,
        geom_net,
        geom_point,
        geom_rink,
        ggplot,
        interactive,
        label_number,
        labs,
        mo,
        p9,
        pl,
        prepare_data,
        to_html,
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
def _(batch_read_shot_data, c, pl):
    shot_data = batch_read_shot_data("/output/shot_data/20252026-wide-window/*.parquet").with_columns(season=pl.lit('20252026'))

    pre_shot_xg = pl.scan_parquet("/output/predictions/*/pre_shot_0809.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/*/post_shot_one_hot_0809.parquet")

    player_mappings = pl.scan_csv("mappings/NHL_20242025_20252026_player_sportlogiq_id_map.csv").with_columns(c('SportlogiqPlayerID').cast(pl.String)).drop('EntityOfficialID')
    return post_shot_xg, pre_shot_xg, shot_data


@app.cell
def _(
    c,
    pl,
    post_shot_xg,
    pre_shot_xg,
    prepare_data,
    reg_season_game_ids,
    shot_data,
):
    data = (
        shot_data
        .filter(c('game_id').is_in(reg_season_game_ids.implode()))
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
        )
        .with_columns(
            pl.when(c('type').str.contains('blocked')).then(0).otherwise(c('post_shot')).alias('post_shot')
        )
        .with_columns(c('post_shot').fill_null(c('pre_shot')))
        .with_columns(
            sga = c('post_shot') - c('pre_shot')
        ).with_columns(
            sga_abs = c('sga').abs()
        ).with_columns(
            shooter_name = pl.concat_str(c('player_first_name', 'player_last_name'), separator=" "),
            tooltip=pl.format(
                "<b>PreXG</b>: {} <br> <b>PostXG</b>: {}",
                c('pre_shot').round(2), c('post_shot').round(2)
            ),
            data_id=pl.concat_str(c('game_id'), pl.lit('s'), c('shot_id'))
        ).collect()
    )
    return (data,)


@app.cell
def _(c, data, mo, pl):
    shooter_names = data.select(pl.concat_str(c('player_first_name', 'player_last_name'), separator=" ").unique()).to_series()

    player_selector = mo.ui.dropdown(shooter_names, searchable=True, allow_select_none=False, value=shooter_names.first())
    return (player_selector,)


@app.cell
def _(c, data, shooter_name):
    shooter_data = (
        data
        .filter(
            c('season') == '20252026',
            c('shooter_name') == shooter_name,
            c('type').str.contains('blocked').not_(),
            c('goalline_y').is_not_null(),
            c('goalline_z').is_not_null(),
        )
    )

    sga = shooter_data.select(c('sga').sum()).item()
    sga_colour = 'green' if sga >= 0 else 'red'
    return (shooter_data,)


@app.cell
def _(player_selector):
    shooter_name = player_selector.value
    subtitle = "xG <added> and <lost> by shot execution (2025-26 Reg. Season)"

    low_colour = "red"
    high_colour = "green"
    mid_colour = "white"
    return high_colour, low_colour, mid_colour, shooter_name


@app.cell
def _(
    aes,
    geom_net,
    geom_point,
    ggplot,
    high_colour,
    label_number,
    labs,
    low_colour,
    mapping,
    mid_colour,
    p9,
    shooter_data,
):
    goal_plot = (
        ggplot(shooter_data, aes(x='goalline_y', y='goalline_z', size='pre_shot', fill='sga'))
        + geom_net()
        + geom_point(mapping=mapping)
        + p9.scale_fill_gradient2(low=low_colour, mid=mid_colour, high=high_colour, labels=label_number(style_positive="+"))
        + p9.scale_size_continuous(breaks=[0.2,0.4,0.6], limits=(0,1), range=(1,9))
        + p9.scale_x_reverse()
        + p9.coord_fixed(ratio=1, ylim=(0, 6), xlim=(6, -6))
        + p9.theme_minimal(base_size=12)
        + labs(x="", y="", fill="SGA", size="PreXG")
        + p9.theme(
            plot_caption=p9.element_text(ha="center"),
            axis_text=p9.element_blank(), axis_ticks=p9.element_blank()
        )
        + p9.guides(
            size=False,
            fill=False
        )
    )
    return (goal_plot,)


@app.cell
def _(aes):
    mapping = aes(tooltip="tooltip", data_id="data_id", hover_key="data_id")
    return (mapping,)


@app.cell
def _(
    aes,
    geom_rink,
    ggplot,
    high_colour,
    low_colour,
    mapping,
    mid_colour,
    p9,
    shooter_data,
):
    rink_plot = (
        ggplot(shooter_data, aes(x="shot_x", y="shot_y", fill="sga"))
        + geom_rink()
        + p9.geom_point(
            mapping=mapping, alpha=0.9, shape="p", size=5, color="black"
            # arrow=p9.arrow(type="closed", angle=15, length=0.1)
        )
        + p9.scale_fill_gradient2(low=low_colour, mid=mid_colour, high=high_colour)
        + p9.coord_fixed(xlim=(25, None))
        + p9.theme_void()
        + p9.guides(fill=False)
    )
    return (rink_plot,)


@app.cell
def _(css, goal_plot, interactive, rink_plot, to_html):
    plot = (goal_plot / rink_plot)

    plot_html = (
        interactive(plot)
        + css(from_dict={".tooltip": {"font-size": "1.1em", "padding": "8px 10px"}})
        + to_html()
    )
    return (plot_html,)


@app.cell
def _(mo, player_selector, plot_html):
    mo.vstack([
        player_selector,
        mo.iframe(plot_html)
    ])
    return


if __name__ == "__main__":
    app.run()
