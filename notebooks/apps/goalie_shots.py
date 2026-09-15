import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, labs
    from mizani.labels import label_number
    from mizani.bounds import squish
    from ninejs import interactive, css, to_html, save

    from data_readers import batch_read_shot_data, read_player_id_mapping
    from models.data import prepare_data
    from plotting.rink import geom_rink, geom_net, geom_ice

    return (
        aes,
        batch_read_shot_data,
        c,
        css,
        geom_ice,
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
        save,
        squish,
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
    return player_mappings, post_shot_xg, pre_shot_xg, shot_data


@app.cell
def _(
    c,
    pl,
    player_mappings,
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
            player_mappings,
            left_on='opposing_team_goalie_on_ice_ref',
            right_on='SportlogiqPlayerID',
            how='left'
        ).rename({'PlayerName': 'opposing_goalie_name'})
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
            gsax = c('post_shot') - c('goal')
        ).with_columns(
            gsax_abs = c('gsax').abs()
        ).with_columns(
            shooter_name = pl.concat_str(c('player_first_name', 'player_last_name'), separator=" "),
            result=(
                pl.when(c('goal')).then(pl.lit('Goal'))
                .when(c('outcome') == 'successful').then(pl.lit('On Target'))
                .otherwise(pl.lit('Missed'))
                .cast(pl.Enum(['Goal', 'On Target', 'Missed']))
            ),
        ).with_columns(
            tooltip=pl.format(
                """
                <b>{}</b> <br>
                <b>PreXG</b>: {} <br>
                <b>PostXG</b>: {} <br>
                <b>Speed</b>: {}mph <br>
                <b>Outcome</b>: {} <br>
                """,
                c('shooter_name'),
                c('pre_shot').round(2),
                c('post_shot').round(2),
                (c('shot_speed') * 0.681818).round(1),
                c('result')
            ),
            data_id=pl.concat_str(c('game_id'), pl.lit('s'), c('shot_id'))
        ).collect()
    )
    return (data,)


@app.cell
def _(c, data, mo):
    goalie_names = data.select(c('opposing_goalie_name').unique()).to_series()

    player_selector = mo.ui.dropdown(goalie_names, searchable=True, allow_select_none=False, value=goalie_names.first())
    return (player_selector,)


@app.cell
def _(player_selector):
    goalie_name = player_selector.value

    low_colour = "red"
    high_colour = "green"
    mid_colour = "white"
    return goalie_name, high_colour, low_colour, mid_colour


@app.cell
def _(c, data, goalie_name):
    goalie_data = (
        data
        .filter(
            c('season') == '20252026',
            c('opposing_goalie_name') == goalie_name,
            c('type').str.contains('blocked').not_(),
            c('goalline_y').is_not_null(),
            c('goalline_z').is_not_null(),
        ).sort(c('result'))
    )

    gsax = goalie_data.select((c('post_shot') - c('goal')).sum()).item()
    gsax_colour = 'green' if gsax >= 0 else 'red'
    return (goalie_data,)


@app.cell
def _(
    aes,
    geom_ice,
    geom_net,
    geom_point,
    ggplot,
    goalie_data,
    goalie_name,
    high_colour,
    label_number,
    labs,
    low_colour,
    mapping,
    mid_colour,
    p9,
    squish,
):
    goal_plot = (
        ggplot(goalie_data, aes(x='goalline_y', y='goalline_z', size='gsax_abs', fill='gsax'))
        + geom_net()
        + geom_ice()
        + geom_point(mapping=mapping, alpha=0.5)
        + p9.scale_fill_gradient2(limits=(-0.6,0.6), low=low_colour, mid=mid_colour, high=high_colour, labels=label_number(style_positive="+"), oob=squish)
        + p9.scale_size_continuous(breaks=[0.2,0.4,0.6], limits=(0,1), range=(1,6))
        + p9.scale_x_reverse()
        + p9.coord_fixed(ratio=1, ylim=(0, 6), xlim=(4.5, -4.5))
        + p9.theme_538()
        + labs(x="", y="", fill="GSAX", size="", alpha="", title=goalie_name, subtitle="Reg. Season Unblocked Shots Faced 2025-26")
        + p9.theme(
            axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
            plot_title=p9.element_text(weight='bold'),
            plot_subtitle=p9.element_text(size=8),
            legend_position='bottom', legend_box='vertical', legend_key_height=12
        ) + p9.guides(
            size=p9.guide_legend(override_aes={'fill': 'white'})
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
    goalie_data,
    high_colour,
    label_number,
    low_colour,
    mapping,
    mid_colour,
    p9,
):
    rink_plot = (
        ggplot(goalie_data, aes(x="shot_x", y="shot_y", fill="gsax", shape='result'))
        + geom_rink(fill='white')
        + p9.geom_point(
            mapping=mapping, size=2, alpha=0.4, color="black"
            # arrow=p9.arrow(type="closed", angle=15, length=0.1)
        )
        + p9.scale_fill_gradient2(limits=(-0.6,0.6), low=low_colour, mid=mid_colour, high=high_colour, labels=label_number(style_positive="+"))
        + p9.scale_shape_manual(values={'Goal': '*', 'Missed': 's', 'On Target': '^'})
        + p9.coord_fixed(xlim=(25, None))
        + p9.theme_538()
        + p9.theme(
            panel_grid=p9.element_blank(),
            axis_title=p9.element_blank(),
            axis_text=p9.element_blank(),
            legend_title=p9.element_blank(),
            legend_position='bottom',
            plot_title=p9.element_blank(),
            plot_subtitle=p9.element_blank(),
        )
        + p9.guides(
            fill=False,
            shape=p9.guide_legend(override_aes={'fill': 'white'})
        )
    )
    return (rink_plot,)


@app.cell
def _(css, goal_plot, interactive, rink_plot, to_html):
    composed = goal_plot | rink_plot

    plot = (
        interactive(composed)
        + css(from_dict={
            ".tooltip": {"font-size": "1.1em", "padding": "8px 10px"},
            ".plot-element.hovered": {"fill": "#6642f5"},
        })
    )

    plot_html = plot + to_html()
    return plot, plot_html


@app.cell
def _(css, plot, save, save_button, shooter_name):
    if save_button.value:
        plot + css(from_dict={
            "svg": {
                "width": "auto", 
                "height": "80vh"
            },
            "div": {"flex-wrap": "wrap", "justify-content": "center"}
        }) + save(f"output/interactive/{shooter_name}.html")
    return


@app.cell
def _(mo, player_selector, plot_html):
    save_button = mo.ui.run_button(label="Save")

    mo.vstack([
        mo.hstack([player_selector, save_button], justify="start"),
        mo.iframe(plot_html)
    ])
    return (save_button,)


if __name__ == "__main__":
    app.run()
