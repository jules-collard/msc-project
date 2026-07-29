import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_violin, labs

    from data_readers import batch_read_shot_data


@app.cell
def _():
    model_data = (
        batch_read_shot_data("/output/shot_data/*.parquet")
        .filter(
            c('shot_x').is_between(25, 89, closed="left"), # Only o-zone shots
            c('opposing_team_goalie_on_ice_ref').is_not_null() # no empty net
        ).collect()
    )

    unblocked = model_data.filter(c('type').str.contains('blocked').not_())
    return model_data, unblocked


@app.cell
def _():
    caption = "O-Zone Shot Attempts Oct.-Dec. 2025"
    unblocked_caption = "Unblocked O-Zone Shots Oct.-Dec. 2025"
    return caption, unblocked_caption


@app.cell
def _(unblocked, unblocked_caption):
    (
        unblocked
        >> ggplot(aes(x='goal', y='goalie_speed', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.coord_flip()
        + p9.theme_bw(base_size=10)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Goaltender Speed by Goal Outcome",
               x="", y="Goaltender Speed (ft/s)", caption=unblocked_caption)
    )
    return


@app.cell
def _(unblocked, unblocked_caption):
    (
        unblocked
        >> ggplot(aes(x='goal', y='lateral_speed', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.coord_flip()
        + p9.theme_bw(base_size=10)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Goaltender Lateral Speed by Goal Outcome",
               x="", y="Goaltender Lateral Speed (ft/s)", caption=unblocked_caption)
    )
    return


@app.cell
def _(unblocked, unblocked_caption):
    (
        unblocked
        >> ggplot(aes(x='goal', y='goalie_dist_to_goal', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.coord_flip()
        + p9.theme_bw(base_size=10)
        + p9.ylim((0, 15))
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Goaltender Distance to Goal by Goal Outcome",
               x="", y="Goaltender Distance to Goal (ft)", caption=unblocked_caption)
    )
    return


@app.cell
def _(model_data, unblocked_caption):
    (
        model_data
        >> ggplot(aes(x='goal', y='goalie_angle_to_shooter', fill='goal'))
        + p9.geom_hline(yintercept=0, linetype='dotted')
        + p9.geom_violin(alpha=0.9, show_legend=False)
        + p9.coord_flip()
        + p9.ylim((-20, 20))
        + p9.theme_bw(base_size=10)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + p9.theme(plot_subtitle=p9.element_text(ha="center"))
        + labs(title="Distribution of Goaltender Angle to Shooter by Goal Outcome",
               subtitle="← Left of Shooting Lane      Right of Shooting Lane →",
               x="", y="Goaltender Angle to Shooter (degrees)",
               caption=unblocked_caption)
    )
    return


@app.cell
def _(caption, model_data):
    (
        model_data
        >> ggplot(aes(x='goal', y='total_pressure', fill='goal'))
        + p9.geom_violin(show_legend=False)
        + p9.coord_flip()
        + p9.theme_bw(base_size=10)
        + p9.scale_x_discrete(labels=["No Goal", "Goal"])
        + labs(title="Distribution of Defensive Pressure by Goal Outcome",
               x="", y="Defensive Pressure", caption=caption)
    )
    return


@app.cell
def _(caption, model_data):
    (
        model_data
        .with_columns(pl.when(c('goal')).then(pl.lit('Goal')).otherwise(pl.lit('No Goal')).alias('goal'))
        >> ggplot(aes(x='num_defenders_in_shooting_lane', y=p9.after_stat('prop'), fill='goal'))
        + p9.geom_bar(show_legend=False)
        + p9.scale_fill_discrete(direction=-1)
        + p9.facet_wrap('goal')
        + p9.theme_bw(base_size=10)
        + p9.labs(title="Distribution of Defenders in Shooting Lane by Goal Outcome",
                 y="Proportion", x="# Defenders in Shooting Lane",
                 caption=caption)
    )
    return


@app.cell
def _(caption, model_data):
    (
        model_data
        .with_columns(pl.when(c('goal')).then(pl.lit('Goal')).otherwise(pl.lit('No Goal')).alias('goal'))
        >> ggplot(aes(x='num_defenders_in_shadow_lane', y=p9.after_stat('prop'), fill='goal'))
        + p9.geom_bar(show_legend=False)
        + p9.scale_fill_discrete(direction=-1)
        + p9.facet_wrap('goal')
        + p9.theme_bw(base_size=10)
        + p9.labs(title="Distribution of Defenders in Shadow Lane by Goal Outcome",
                 y="Proportion", x="# Defenders in Shadow Lane",
                 caption=caption)
    )
    return


if __name__ == "__main__":
    app.run()
