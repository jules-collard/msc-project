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
    from highlight_text import ax_text, fig_text
    from matplotlib import pyplot as plt
    import matplotlib.font_manager

    from data_readers import batch_read_shot_data
    from models.data import prepare_data

    return (
        aes,
        ax_text,
        batch_read_shot_data,
        c,
        geom_point,
        ggplot,
        label_number,
        labs,
        mo,
        p9,
        pl,
        prepare_data,
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
def _(c, data, mo):
    goalie_names = data.select(c('opposing_goaltender_name').unique()).to_series()

    goalie_selector = mo.ui.dropdown(
        goalie_names,
        searchable=True
    )
    return (goalie_selector,)


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
        .with_columns(
            pl.when(c('type').str.contains('blocked')).then(0).otherwise(c('post_shot')).alias('post_shot')
        )
        .with_columns(c('post_shot').fill_null(c('pre_shot')))
        .with_columns(
            gsax = c('post_shot') - c('goal')
        ).with_columns(
            gsax_abs = c('gsax').abs()
        )
        .collect()
    )
    return (data,)


@app.cell
def _(c, data, goalie_name):
    goalie_data = (
        data
        .filter(
            c('season') == '20252026',
            c('opposing_goaltender_name') == goalie_name,
            c('type').str.contains('blocked').not_(),
            c('goalline_y').is_not_null(),
            c('goalline_z').is_not_null(),
        )
    )

    gsax = goalie_data.select(c('gsax').sum()).item()
    gsax_colour = 'green' if gsax >= 0 else 'red'
    return goalie_data, gsax, gsax_colour


@app.cell
def _(goalie_selector):
    goalie_name = goalie_selector.value
    subtitle = "Unblocked shots <saved> and <conceded> (2025-26 Reg. Season)"
    caption = "← Blocker Side       Glove Side →"

    low_colour = "red"
    high_colour = "green"
    mid_colour = "white"
    return caption, goalie_name, high_colour, low_colour, mid_colour, subtitle


@app.cell
def _(aes, p9, pl):
    net_outline = pl.DataFrame({
        'y': [-3, -3, 3, 3],
        'z': [0, 4, 4, 0]
    })

    net = p9.geom_rect(
        aes(xmin=-3, xmax=3, ymin=0, ymax=4, alpha=0.8), fill='lightgrey', show_legend=False
    )

    posts = p9.geom_path(
        aes(x='y', y='z'), 
        data=net_outline, 
        color="red", size=2, lineend="round"
    )
    return net, posts


@app.cell
def _(
    aes,
    ax_text,
    caption,
    geom_point,
    ggplot,
    goalie_data,
    goalie_name,
    gsax,
    gsax_colour,
    high_colour,
    label_number,
    labs,
    low_colour,
    mid_colour,
    net,
    p9,
    posts,
    subtitle,
):
    p = (
        ggplot()
        + net
        + posts
        + geom_point(
            aes(x='goalline_y_norm', y='goalline_z', size='gsax_abs', fill='gsax', alpha='gsax_abs'),
            data=goalie_data,
        ) + p9.geom_label(
            aes(x=3, y=5), color=gsax_colour, label=f"GSAX: {gsax:.2f}"
        )
        + p9.scale_fill_gradient2(low=low_colour, mid=mid_colour, high=high_colour, labels=label_number(style_positive="+"))
        + p9.scale_size_continuous(breaks=[0.25, 0.5, 0.75], labels=lambda x: [f"±{label}" for label in x], limits=(0,1))
        + p9.scale_alpha_continuous(breaks=[0.25, 0.5, 0.75], labels=lambda x: [f"±{label}" for label in x], limits=(0,1))
        + p9.scale_x_reverse()
        + p9.coord_fixed(ratio=1, ylim=(0, 5), xlim=(4, -4))
        + p9.theme_538(base_size=12)
        + labs(x="", y="", fill="GSAX", alpha="", size="",
               title=" ", subtitle=" ",
               caption=caption)
        + p9.theme(
            plot_caption=p9.element_text(ha="center"),
            axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
            dpi=300
        )
        + p9.guides(
            alpha=p9.guide_legend(override_aes={'fill':'white'}),
            fill=p9.guide_colorbar(theme=p9.theme(legend_key_width=12, legend_key_height=80))
        )
    )

    fig = p.draw()
    ax = fig.axes[0]

    ax_text(
        s=f"<{goalie_name}>\n{subtitle}",
        x=-4.2, y=5.5, ax=ax, va='bottom',
        highlight_textprops=[
            {'fontsize': 18},
            {'color': high_colour, 'fontweight': 'bold'}, 
            {'color': low_colour, 'fontweight': 'bold'}, 
        ]
    )

    None
    return ax, fig


@app.cell
def _(fig, goalie_name, save_button):
    if save_button.value:
        fig.savefig(f"plots/goalie_plots/{goalie_name}.png", dpi=300, bbox_inches='tight')
    return


@app.cell
def _(ax, goalie_selector, mo):
    save_button = mo.ui.run_button(label="Save")

    mo.vstack([
        mo.hstack([goalie_selector, save_button], justify="start"),
        ax
    ])
    return (save_button,)


if __name__ == "__main__":
    app.run()
