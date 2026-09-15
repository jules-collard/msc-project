import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    from datetime import date

    import marimo as mo
    import polars as pl
    from polars import col as c
    import polars.selectors as cs
    import plotnine as gg
    from plotnine import ggplot, aes
    import mizani.labels as ml
    from ninejs import interactive, save, css

    from models.data import prepare_data, post_shot_filter
    from models.features import post_shot_feature_groups
    from data_readers import read_game_id_mapping, batch_read_shot_data


@app.cell
def _():
    game_ids = (
        read_game_id_mapping("mappings/NHL_20242025_20252026_game_smt_sportlogiq_id_map.csv")
        .filter(
            c('Stage') == 'regular',
            c('GameDate') >= date(2025, 10, 1),
            (c('AwayTeam') == 'MTL') | (c('HomeTeam') == 'MTL')
        ).select(c('SportlogiqGameID').cast(pl.String))
        .to_series().to_list()
    )
    return (game_ids,)


@app.cell
def _(game_ids):
    shot_data = batch_read_shot_data([f"/output/shot_data/20252026-wide-window/{id}_shot_data.parquet" for id in game_ids])

    shap = (
        pl.scan_parquet([f"/output/shap/20252026/{id}_shap.parquet" for id in game_ids])
    )

    data = (
        shot_data
        .with_columns(pl.concat_str(c('player_first_name'), c('player_last_name'), separator=" ").alias('player_name'))
        .pipe(prepare_data)
        .pipe(post_shot_filter)
        .filter(c('team') == 'Montreal Canadiens')
        .select(c('game_id', 'period', 'shot_id', 'player_reference_id', 'player_name'))   
        .join(
            shap,
            on=['game_id', 'period', 'shot_id'],
            how='inner'
        ).with_columns(
            pl.sum_horizontal(*post_shot_feature_groups.keys()).alias('sum_shap')
        )
    )
    return (data,)


@app.cell
def _(data):
    formatted_shap = (
        pl.when(c('shap') > 0)
        .then(pl.format("+{}", c('shap').round(2))) # Add '+' if positive
        .otherwise(c('shap').round(2).cast(pl.String)) # Negatives get '-' automatically
    )

    plot_data = (
        data
        .with_columns(
            post_shot = pl.sum_horizontal('base_value', *post_shot_feature_groups.keys()),
        ).group_by('player_reference_id', 'player_name')
        .agg(
            c('base_value', *post_shot_feature_groups.keys(), 'post_shot').sum()
        )
        .sort(c('post_shot'), descending=True)
        .head(8)
        .unpivot(
            on=post_shot_feature_groups.keys(),
            index=['player_reference_id', 'player_name', 'post_shot'],
            variable_name='variable',
            value_name='shap'
        ).with_columns(
            c('variable').str.replace("_", " ").str.to_titlecase(),
        ).with_columns(
            tooltip = pl.format("""
                <b>{}</b> <br>
                {} xG
            """, c('variable'), formatted_shap
            )
        )
        .collect()
    )
    return (plot_data,)


@app.cell
def _():
    labels = [f"<{label}>" for label in post_shot_feature_groups.keys()]
    colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']

    contrib_string = ", ".join(labels).replace("_", " ")
    return


@app.cell
def _(plot_data):
    p = (
        ggplot(plot_data, aes(x='reorder(player_name, post_shot)', y='shap'))
        + gg.geom_col(aes(fill='variable', tooltip='tooltip', hover_group='variable'))
        # + gg.geom_label(aes(label='post_shot', y='shap_change + 3'), format_string='{:.2f}')
        + gg.geom_hline(yintercept=0)
        + gg.scale_y_continuous(labels=ml.label_number(style_positive="+"))
        + gg.scale_fill_brewer(type="qual", palette="Paired")
        + gg.labs(
            x="", y="xG Added/Lost",
            title="Deconstructing Post-Shot xG",
            subtitle="Top 8 Montreal Canadiens by Post-Shot xG, 2025-26 Reg. Season"
        )
        + gg.theme_538()
        + gg.coord_flip()
        + gg.theme(
            panel_grid_major_y=gg.element_blank(),
            legend_title=gg.element_blank(),
            plot_title_position="plot",
            plot_title=gg.element_text(weight='bold'),
            plot_subtitle=gg.element_text(size=10),
            dpi=300
            # axis_text_x=gg.element_text(angle=45)
        )
    )

    (
        interactive(p)
        + css(from_dict={
            ".tooltip": {"font-size": "1.1em", "padding": "8px 10px"},
            ".plot-element.hovered": {"stroke": "#28282B"},
        })
        # + css(from_dict={
        #     "svg": {
        #         "width": "auto", 
        #         "height": "80vh"
        #     },
        #     "div": {"flex-wrap": "wrap", "justify-content": "center"}
        # }) + save(f"output/interactive/Canadiens.html")
    )
    return


@app.cell
def _(plot_data):
    p_static = (
        ggplot(plot_data, aes(x='reorder(player_name, post_shot)', y='shap'))
        + gg.geom_col(aes(fill='variable'))
        # + gg.geom_label(aes(label='post_shot', y='shap_change + 3'), format_string='{:.2f}')
        + gg.geom_hline(yintercept=0)
        + gg.scale_y_continuous(labels=ml.label_number(style_positive="+"))
        + gg.scale_fill_brewer(type="qual", palette="Paired")
        + gg.labs(
            x="", y="xG Added/Lost",
            title="Deconstructing Post-Shot xG",
            subtitle="Top 8 Montreal Canadiens by Post-Shot xG, 2025-26 Reg. Season"
        )
        + gg.theme_538()
        + gg.coord_flip()
        + gg.theme(
            panel_grid_major_y=gg.element_blank(),
            legend_title=gg.element_blank(),
            plot_title_position="plot",
            plot_title=gg.element_text(weight='bold'),
            plot_subtitle=gg.element_text(size=10),
            dpi=300
            # axis_text_x=gg.element_text(angle=45)
        )
    )

    # p_static.save("output/viz/post_shot_breakdown.png")
    p_static
    return


if __name__ == "__main__":
    app.run()
