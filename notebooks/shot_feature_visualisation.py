import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, labs, theme_bw
    from mizani.labels import percent_format

    from data_readers import batch_read_events, batch_read_puck_tracking
    from post_shot.features import PostShotData
    from interaction import game_selectors, display_game_selectors
    from utils import distance_to_point_2d

    return (
        aes,
        batch_read_events,
        batch_read_puck_tracking,
        c,
        cs,
        display_game_selectors,
        distance_to_point_2d,
        game_selectors,
        ggplot,
        labs,
        mo,
        p9,
        percent_format,
        pl,
        theme_bw,
    )


@app.cell(disabled=True)
def _(c, pl):
    game_id_mapping = (
        pl.read_csv("mappings/NHL_20252026_game_smt_sportlogiq_id_map.csv")
        .with_columns(c('GameDate').str.strptime(pl.Date, format="%Y-%m-%d"))
    )
    return (game_id_mapping,)


@app.cell(disabled=True)
def _(display_game_selectors, game_id_mapping, game_selectors):
    date_selector, game_id_selector, game_type_selector, run_button = game_selectors(game_id_mapping)
    display_game_selectors(date_selector, game_id_selector, game_type_selector, run_button)
    return date_selector, game_id_selector, game_type_selector, run_button


@app.cell(disabled=True)
def _(
    c,
    date_selector,
    game_id_mapping,
    game_id_selector,
    game_type_selector,
    pl,
):
    games = (
        game_id_mapping
        .filter(
            c('GameDate').is_in(pl.date_range(*date_selector.value).implode()),
            c('SportlogiqGameID').is_in(game_id_selector.value),
            c('Stage').is_in(game_type_selector.value)
        )
    )

    sportlogiq_ids = games.select(c('SportlogiqGameID')).to_series().to_list()
    SMT_ids = games.select(c('SMTGameID')).to_series().to_list()
    return SMT_ids, games, sportlogiq_ids


@app.cell(disabled=True)
def _(
    SMT_ids,
    batch_read_events,
    batch_read_puck_tracking,
    games,
    mo,
    run_button,
    sportlogiq_ids,
):
    mo.stop(not run_button.value, mo.md("Press **Run Shot Detection** above to load the data and generate the plots."))


    events = batch_read_events([f"data/sportlogiq/*/games/{id}/*_sapifullevents.json" for id in sportlogiq_ids])

    puck_tracking = batch_read_puck_tracking(
        [f"data/smtoasis/*/games/{id}/*_puck_tracking_raw_measurements*.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )
    return


@app.cell
def _(c, distance_to_point_2d, pl):
    # post_shot_data = PostShotData(events, puck_tracking)
    # model_data = post_shot_data.model_data().collect()
    model_data = (
        pl.read_parquet("/output/post_shot_data_202510.parquet")
        .drop_nulls(c('shot_speed')) # Only keep shots with valid speed (i.e. shots that were detected)
        .filter(
            c('shot_x') >= 25, # Only o-zone shots
            c('type').str.contains('blocked').not_()
        ).with_columns(
            dist_to_goal = distance_to_point_2d(c('shot_x'), c('shot_y'), 89, 0),
            goalline_y_norm = pl.when(
                c('goalie_handedness') == 'R'
            ).then(
                -c('goalline_y')
            ).otherwise(c('goalline_y'))
        )
    )
    return (model_data,)


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='shot_speed', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.coord_flip()
        + labs(title="Distribution of Shot Speed by Goal Outcome",
              x="", y="Shot Speed (ft/s)")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='shot_speed'))
        + p9.geom_sina(aes(fill='dist_to_goal', color='dist_to_goal'), alpha=0.2)
        + p9.geom_violin(color='black', fill=None, alpha=0.7)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.scale_fill_distiller(palette='YlOrRd', direction=-1)
        + p9.scale_color_distiller(palette='YlOrRd', direction=-1)
        + p9.coord_flip()
        + labs(title="Distribution of Shot Speed by Goal Outcome",
              x="", y="Shot Speed (ft/s)", fill="Distance \n to Goal (ft)",
              color="Distance \n to Goal (ft)")
        + theme_bw()
    )
    return


@app.cell
def _(aes, c, ggplot, labs, model_data, p9, percent_format, pl, theme_bw):
    grid_size=1

    net_outline = pl.DataFrame({
        'y': [-3, -3, 3, 3],
        'z': [0, 4, 4, 0]
    })

    (
        model_data
        .with_columns(
            ((pl.col('goalline_y') / grid_size).floor() * grid_size + (grid_size / 2)).alias('grid_y'),
            ((pl.col('goalline_z') / grid_size).floor() * grid_size + (grid_size / 2)).alias('grid_z')
        ).group_by('grid_y', 'grid_z')
        .agg(
            c('goal').mean().alias('success_rate'),
            pl.len().alias('num_shots')
        ).filter(
            c('grid_y').is_between(-5, 5),
            c('grid_z').is_between(0, 6),
        )
        >> ggplot(aes(x='grid_y', y='grid_z'))
        + p9.geom_tile(aes(fill='success_rate'))
        + p9.geom_point(aes(size='num_shots'), fill='white', color='black', alpha=0.7)
        # Ice Surface
        + p9.geom_segment(
            aes(x=-5, xend=5, y=0, yend=0), 
            color="lightblue", size=2
        ) 
        # Posts
        + p9.geom_path(
            aes(x='y', y='z'), 
            data=net_outline, 
            color="red", size=2, lineend="round"
        )
        + p9.coord_fixed(ratio=1, ylim=(0, 6))
        + p9.scale_x_reverse()
        + p9.scale_fill_continuous(labels=percent_format())
        + p9.scale_size_continuous()
        + theme_bw()
        + labs(title="Shot Success Rate by Goal Location",
              x="Horizontal Location (ft)", y="Vertical Location (ft)",
              fill="Success Rate", size="# Shots",
              caption="Unblocked Shots")
    )
    return grid_size, net_outline


@app.cell
def _(
    aes,
    c,
    ggplot,
    grid_size,
    labs,
    model_data,
    net_outline,
    p9,
    percent_format,
    pl,
    theme_bw,
):
    mean_success_rate = model_data.select(c('goal').mean()).item()

    (
        model_data
        .with_columns(
            ((pl.col('goalline_y_norm') / grid_size).floor() * grid_size + (grid_size / 2)).alias('grid_y'),
            ((pl.col('goalline_z') / grid_size).floor() * grid_size + (grid_size / 2)).alias('grid_z')
        ).group_by('grid_y', 'grid_z')
        .agg(
            (c('goal').mean() - mean_success_rate).alias('rel_success_rate'),
            pl.len().alias('num_shots')
        ).filter(
            c('grid_y').is_between(-5, 5),
            c('grid_z').is_between(0, 6),
        )
        >> ggplot(aes(x='grid_y', y='grid_z'))
        + p9.geom_tile(aes(fill='rel_success_rate'))
        + p9.geom_point(aes(size='num_shots'), fill='white', color='black', alpha=0.7)
        + p9.geom_segment(
            aes(x=-5, xend=5, y=0, yend=0), 
            color="lightblue", size=2
        ) 
        # Posts
        + p9.geom_path(
            aes(x='y', y='z'), 
            data=net_outline, 
            color="black", size=2, lineend="round"
        ) + p9.annotate("text", label="← Blocker Side", x=1.5, y=7.2,
                        size=10, color="black")
        + p9.annotate("text", label="Glove Side →", x=-1.5, y=7.2,
                      size=10, color="black")
        + p9.coord_fixed(ratio=1, ylim=(0, 6))
        + p9.scale_x_reverse()
        + p9.scale_fill_cmap(cmap_name="bwr", limits=(-0.15, 0.15), labels=percent_format())
        + p9.scale_size_continuous()
        + theme_bw()
        + p9.theme(plot_subtitle=p9.element_text(ha="center"))
        + labs(title="Relative Shot Success Rate by Goal Location",
               subtitle="← Blocker Side       Glove Side →",
              x="Horizontal Location (ft)", y="Vertical Location (ft)",
              fill="Success Rate", size="# Shots",
              caption=f"Unblocked Shots | Success Rate relative to League Average {mean_success_rate:.1%}")
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='goalline_y', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=3, linetype="dotted")
        + p9.geom_hline(yintercept=-3, linetype="dotted")
        + p9.scale_y_reverse()
        + p9.ylim(25, -25)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.coord_flip()
        + labs(title="Horizontal Distribution of Shots by Goal Outcome",
              x="", y="Horizontal Goalline Location (ft)")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='goalline_z', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=4, linetype="dotted")
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.ylim(0, 10)
        + labs(title="Vertical Distribution of Shots by Goal Outcome",
              x="", y="Vertical Goalline Location (ft)")
        + p9.theme(legend_position="none")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='dist_to_corner', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Shot Distance to Nearest Corner by Goal Outcome",
              x="", y="Distance to Nearest Corner (ft)")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='dist_to_post', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.annotate("text", label="← Off Target", x=1.4, y=-2,
                     alpha=0.8, size=10)
        + p9.annotate("text", label="On Target →", x=1.4, y=1.5,
                 alpha=0.8, size=10)
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.ylim(-10, None)
        + labs(title="Distribution of Shot Distance to Nearest Post by Goal Outcome",
              x="", y="Distance to Nearest Post (ft)")
        + p9.theme(legend_position="none")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='dist_to_center', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.ylim(0, 20)
        + labs(title="Distribution of Shot Distance to Goal Center by Goal Outcome",
              x="", y="Distance to Goal Center (ft)")
        + p9.theme(legend_position="none")
        + theme_bw()
    )
    return


@app.cell
def _(aes, c, cs, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        .select(c('goal'), cs.starts_with('polar_angle'))
        .unpivot(on=cs.starts_with('polar_angle'), index='goal', value_name='angle', variable_name='angle_type')
        .with_columns(c('angle_type').replace({
            'polar_angle_abs': 'Symmetric',
            'polar_angle_near_post': 'Near Post',
            'polar_angle_raw': 'Static'
        }))
        >> ggplot(aes(x='goal', y='angle', fill='goal'))
        + p9.geom_violin(show_legend=False)
        # + p9.geom_jitter(alpha=0.2, show_legend=False)
        + p9.facet_wrap('angle_type')
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Shot Angles by Reference Point",
              x="", y="Angle (Degrees)")
        + p9.theme(legend_position="none")
        + theme_bw()
    )
    return


if __name__ == "__main__":
    app.run()
