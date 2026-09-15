import marimo

__generated_with = "0.24.2"
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
    from ninejs import interactive

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
            shooter_name = pl.concat_str(c('player_first_name', 'player_last_name'), separator=" ")
        ).collect()
    )
    return (data,)


@app.cell
def _(c, data, mo, pl):
    shooter_names = data.select(pl.concat_str(c('player_first_name', 'player_last_name'), separator=" ").unique()).to_series()

    player_selector = mo.ui.dropdown(shooter_names, searchable=True)
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
    return sga, sga_colour, shooter_data


@app.cell
def _(player_selector):
    shooter_name = player_selector.value
    subtitle = "xG <added> and <lost> by unblocked shot execution (2025-26 Reg. Season)"

    low_colour = "red"
    high_colour = "green"
    mid_colour = "white"
    return high_colour, low_colour, mid_colour, shooter_name, subtitle


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
    geom_point,
    ggplot,
    high_colour,
    label_number,
    labs,
    low_colour,
    mid_colour,
    net,
    p9,
    posts,
    sga,
    sga_colour,
    shooter_data,
    shooter_name,
    subtitle,
):
    p = (
        ggplot()
        + net
        + posts
        + geom_point(
            aes(x='goalline_y', y='goalline_z', size='pre_shot', fill='sga'),
            data=shooter_data,
        ) + p9.geom_label(
            aes(x=4.5, y=5.5), color=sga_colour, label=f"SGA: {sga:.2f}"
        )
        + p9.scale_fill_gradient2(low=low_colour, mid=mid_colour, high=high_colour, labels=label_number(style_positive="+"))
        + p9.scale_size_continuous(breaks=[0.2,0.4,0.6], limits=(0,1), range=(1,9))
        + p9.scale_x_reverse()
        + p9.coord_fixed(ratio=1, ylim=(0, 6), xlim=(6, -6))
        + p9.theme_538(base_size=12)
        + labs(x="", y="", fill="SGA", size="PreXG",
               title=" ", subtitle=" ")
        + p9.theme(
            plot_caption=p9.element_text(ha="center"),
            axis_text=p9.element_blank(), axis_ticks=p9.element_blank(),
            dpi=300, figure_size=(8,4.5)
        )
        + p9.guides(
            size=p9.guide_legend(override_aes={'fill': 'white'}),
            fill=p9.guide_colorbar(theme=p9.theme(legend_key_width=12, legend_key_height=80))
        )
    )

    fig = p.draw()
    ax = fig.axes[0]

    ax_text(
        s=f"<{shooter_name}>\n{subtitle}",
        x=-6.2, y=6.5, ax=ax, va='bottom',
        highlight_textprops=[
            {'fontsize': 18},
            {'color': high_colour, 'fontweight': 'bold'}, 
            {'color': low_colour, 'fontweight': 'bold'}, 
        ]
    )

    None
    return ax, fig


@app.cell
def _(fig, save_button, shooter_name):
    if save_button.value:
        fig.savefig(f"output/viz/{shooter_name}.png", dpi=300, bbox_inches='tight')
    return


@app.cell
def _(ax, mo, player_selector):
    save_button = mo.ui.run_button(label="Save")

    mo.vstack([
        mo.hstack([player_selector, save_button], justify="start"),
        ax
    ])
    return (save_button,)


if __name__ == "__main__":
    app.run()
