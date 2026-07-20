import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, labs, theme_bw
    from mizani.labels import percent_format

    from data_readers import batch_read_events, batch_read_puck_tracking
    from post_shot.features import PostShotData

    return (
        PostShotData,
        aes,
        batch_read_events,
        batch_read_puck_tracking,
        c,
        cs,
        ggplot,
        labs,
        p9,
        percent_format,
        pl,
        theme_bw,
    )


@app.cell
def _(batch_read_events, batch_read_puck_tracking):
    events = batch_read_events(
        "data/*/*_sapifullevents.json"
    )

    puck_tracking = batch_read_puck_tracking(
        "data/*/HOCKEY_NHL_*.parquet"
    )
    return events, puck_tracking


@app.cell
def _(PostShotData, events, puck_tracking):
    post_shot_data = PostShotData(events, puck_tracking)
    model_data = post_shot_data.model_data().collect()
    return (model_data,)


@app.cell
def _(model_data):
    model_data.drop('flags').write_csv('data/post_shot_data_sample.csv')
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='shot_speed', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.coord_flip()
        + labs(title="Distribution of Shot Speed by Goal Outcome",
              x="", y="Shot Speed (ft/s)")
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
        # Net
        # + p9.geom_rect(
        #     aes(xmin=-3, xmax=3, ymin=0, ymax=4), fill='lightgrey',
        # )
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
        + p9.scale_size_continuous(breaks=[5,10,15,20])
        + theme_bw()
        + labs(title="Shot Success Rate by Goal Location",
              x="Horizontal Location (ft)", y="Vertical Location (ft)",
              fill="Success Rate", size="# Shots")
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='goalline_y', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.geom_sina(alpha=0.2, show_legend=False)
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
        + p9.geom_sina(alpha=0.2, show_legend=False)
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
        >> ggplot(aes(x='goal', y='dist_to_top_corner', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Shot Distance to Top Corner by Goal Outcome",
              x="", y="Distance to Nearest Top Corner (ft)")
        + theme_bw()
    )
    return


@app.cell
def _(aes, ggplot, labs, model_data, p9, theme_bw):
    (
        model_data
        >> ggplot(aes(x='goal', y='dist_to_post', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.geom_sina(alpha=0.2, show_legend=False)
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.coord_flip()
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
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
        + p9.geom_sina(alpha=0.2, show_legend=False)
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
        + p9.geom_jitter(alpha=0.2, show_legend=False)
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
