import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_violin, labs
    from great_tables import GT, html, md

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
    lane_summary = (
        model_data
        .group_by(c('num_defenders_in_shooting_lane'))
        .agg(
            c('goal').mean().alias('success_rate'),
            pl.len().alias('num_shots')
        ).sort(c('num_defenders_in_shooting_lane'))
    )

    (
        GT(lane_summary)
        .tab_header("Shot Success Rates by No. of Defenders in Shooting Lane")
        .tab_source_note(caption)
        .cols_label(
            num_defenders_in_shooting_lane=html("# Defenders in <br> Shooting Lane"),
            success_rate = "Success Rate",
            num_shots = "# Shots"
        ).fmt_percent(columns="success_rate")
        .data_color(columns="success_rate", palette="viridis", domain=[0, 0.1])
    )
    return


@app.cell
def _(caption, model_data):
    shadow_summary = (
        model_data
        .group_by(c('num_defenders_in_shadow_lane'))
        .agg(
            c('goal').mean().alias('success_rate'),
            pl.len().alias('num_shots')
        ).sort(c('num_defenders_in_shadow_lane'))
    )

    (
        GT(shadow_summary)
        .tab_header("Shot Success Rates by No. of Defenders in Shadow Lane")
        .tab_source_note(caption)
        .cols_label(
            num_defenders_in_shadow_lane=html("# Defenders in <br> Shadow Lane"),
            success_rate = "Success Rate",
            num_shots = "# Shots"
        ).fmt_percent(columns="success_rate")
        .data_color(columns="success_rate", palette="viridis", domain=[0, 0.1])
    )
    return


if __name__ == "__main__":
    app.run()
