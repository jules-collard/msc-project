import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import joblib

    import marimo as mo
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_smooth, geom_line, theme_bw, labs
    import polars as pl
    from polars import col as c
    import polars.selectors as cs
    import shap
    from great_tables import GT

    from data_readers import batch_read_shot_data
    from models.data import prepare_data, polars_to_pandas
    from models.features import pre_shot_features_minimal


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026-clean/*.parquet")
    return (shot_data,)


@app.cell
def _():
    pre_shot_model = joblib.load("models/pre_shot_minimal/pre_shot_minimal_model_2008_s3870.joblib")
    return


@app.cell
def _(shot_data):
    data = (
        shot_data
        .pipe(prepare_data)
        .collect()
    )
    return (data,)


@app.cell
def _(data):
    X = polars_to_pandas(data.select(pre_shot_features_minimal))
    # explainer = shap.TreeExplainer(pre_shot_model.estimator_)
    # shap_values = explainer(X)
    # joblib.dump(shap_values, "models/pre_shot_minimal/shap_values_20252026.joblib")
    shap_values = joblib.load("models/pre_shot_minimal/shap_values_20252026.joblib")
    return (shap_values,)


@app.cell
def _(data, shap_values):
    shap_data_df = data.select(pre_shot_features_minimal)
    shap_values_df = pl.DataFrame(shap_values.values, schema=[f"{feature}_shap" for feature in pre_shot_features_minimal])
    base_values_df = pl.DataFrame(shap_values.base_values, schema=["base_value"])

    shap_df = (
        pl.concat([shap_data_df, shap_values_df, base_values_df], how="horizontal", strict=True)
    )
    return (shap_df,)


@app.cell
def _():
    features_dict = {
      "total_pressure": "Defensive Pressure",
      "num_defenders_in_shadow_lane": "# Defenders in Shadow Lane",
      "goalie_angle_to_shooter": "Goalie Angle to Shooter (degrees)",
      "goalie_in_shooting_lane": "Goalie in Shooting Lane",
      "goalie_dist_to_goal": "Goalie Distance to Goal (ft)",
      "goalie_lateral_speed": "Goalie Lateral Speed (ft/s)",
      "shooter_goal_speed": "Shooter Goalwards Speed (ft/s)",
      "shooter_dist_to_goal": "Shooter Distance to Goal (ft)",
      "shooter_angle_to_goal": "Shooter Angle to Goal (degrees)",
      "visible_angle": "Visible Angle (degrees)",
      "shot_type": "Shot Type"
    }
    return (features_dict,)


@app.cell
def _():
    custom_theme = theme_bw(base_size=10) + p9.theme(legend_title=p9.element_text(angle=90, size=8), legend_title_position="right", legend_key_width=10, legend_text=p9.element_text(size=6))
    return (custom_theme,)


@app.cell
def _(custom_theme, features_dict, shap_values):
    def shap_dependence_plot(df: pl.DataFrame, variable: str):
        interaction_col = df.columns[shap.utils.potential_interactions(shap_values[:,variable], shap_values)[0]]

        aes_y = f"{variable}_shap"
        x_label = features_dict[variable]
        colour_label = features_dict[interaction_col]

        plot = (
            ggplot(df, aes(x=variable, y=aes_y, colour=interaction_col))
            + p9.geom_hline(yintercept=0, linetype="dotted")
            + geom_point(alpha=0.05)
            + labs(x=x_label, y="SHAP", colour=colour_label)
            + custom_theme
        )
        return plot

    return (shap_dependence_plot,)


@app.cell
def _(shap_df):
    plot_data = shap_df.with_columns(c('num_defenders_in_shadow_lane', 'goalie_in_shooting_lane').cast(pl.String))
    return (plot_data,)


@app.cell
def _(shap_dependence_plot, shap_df):
    distance_plot = shap_dependence_plot(shap_df, "shooter_dist_to_goal")
    angle_plot = shap_dependence_plot(shap_df, "shooter_angle_to_goal")
    visible_angle_plot = shap_dependence_plot(shap_df, "visible_angle")
    goal_speed_plot = shap_dependence_plot(shap_df, "shooter_goal_speed")

    shooter_plots = (distance_plot | angle_plot) / (visible_angle_plot | goal_speed_plot)

    # shooter_plots.save("plots/interpretation/shooter_plots_shap.png", width=6, height=6, dpi=500)
    shooter_plots
    return


@app.cell
def _(custom_theme, plot_data, shap_dependence_plot, shap_df):
    goalie_distance_plot = shap_dependence_plot(shap_df, "goalie_dist_to_goal") + p9.xlim(0, 15)
    goalie_lateral_speed_plot = shap_dependence_plot(shap_df, "goalie_lateral_speed")
    goalie_angle_plot = shap_dependence_plot(shap_df, "goalie_angle_to_shooter")

    goalie_lane_plot_data = (
        plot_data
        .with_columns(
            c('goalie_in_shooting_lane').cast(pl.String).fill_null('Missing').str.to_titlecase()
        )
    )

    goalie_lane_plot = (
        ggplot(goalie_lane_plot_data, aes(x="goalie_in_shooting_lane", y="goalie_in_shooting_lane_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1, random_state=54)
        + labs(x="Goalie in Shooting Lane", y="SHAP", colour="Shooter Distance to Goal")
        + custom_theme
    )

    goalie_plots = (goalie_angle_plot | goalie_distance_plot) / (goalie_lateral_speed_plot | goalie_lane_plot)
    goalie_plots.save("plots/interpretation/goalie_plots_shap.png", width=6, height=6, dpi=500)
    goalie_plots
    return


@app.cell
def _(custom_theme, plot_data, shap_dependence_plot, shap_df):
    defender_lane_plot = (
        ggplot(plot_data, aes(x="num_defenders_in_shadow_lane", y="num_defenders_in_shadow_lane_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1, random_state=54)
        + labs(x="Defenders in Shadow Lane", y="SHAP", colour="Distance to Goal")
        + custom_theme
    )

    pressure_plot = shap_dependence_plot(shap_df, "total_pressure")

    shot_type_plot = (
        ggplot(plot_data, aes(x="shot_type", y="shot_type_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1, random_state=54)
        + labs(x="Shot Type", y="SHAP", colour="Distance to Goal")
        + custom_theme
        + p9.theme(axis_text_x=p9.element_text(rotation=-45))
    )

    defender_plots = (defender_lane_plot | pressure_plot) / shot_type_plot
    # defender_plots.save("plots/interpretation/defender_plots_shap.png", width=6, height=6, dpi=500)
    defender_plots
    return


if __name__ == "__main__":
    app.run()
