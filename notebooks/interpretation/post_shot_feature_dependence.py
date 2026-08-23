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
    from models.data import prepare_data, polars_to_pandas, post_shot_filter
    from models.features import post_shot_features_minimal


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026-clean/*.parquet")
    return (shot_data,)


@app.cell
def _():
    model = joblib.load("models/post_shot_minimal/post_shot_minimal_model_2108_s219.joblib")
    return


@app.cell
def _(shot_data):
    data = (
        shot_data
        .pipe(prepare_data)
        .pipe(post_shot_filter)
        .collect()
    )
    return (data,)


@app.cell
def _(data):
    X = polars_to_pandas(data.select(post_shot_features_minimal))
    # explainer = shap.TreeExplainer(model.estimator_)
    # shap_values = explainer(X)
    # joblib.dump(shap_values, "models/post_shot_minimal/shap_values_20252026.joblib")
    shap_values = joblib.load("models/post_shot_minimal/shap_values_20252026.joblib")
    return (shap_values,)


@app.cell
def _(data, shap_values):
    shap_data_df = data.select(post_shot_features_minimal)
    shap_values_df = pl.DataFrame(shap_values.values, schema=[f"{feature}_shap" for feature in post_shot_features_minimal])
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
      "shot_type": "Shot Type",
        "shot_speed": "Shot Speed (ft/s)",
        "goalline_y_norm": "Projected Y Coordinate",
        "goalline_z": "Projected Z Coordinate",
        "on_goal": "On Goal",
        "dist_to_post": "Distance to Nearest Post",
        "dist_to_corner": "Distance to Nearest Corner",
        "dist_to_center": "Distance to Center of Goal"
    }
    return (features_dict,)


@app.cell
def _(features_dict, shap_values):
    def shap_dependence_plot(df: pl.DataFrame, variable: str, limits=None):
        interaction_col = df.columns[shap.utils.potential_interactions(shap_values[:,variable], shap_values)[0]]

        aes_y = f"{variable}_shap"
        x_label = features_dict[variable]
        colour_label = features_dict[interaction_col]

        plot = (
            ggplot(df, aes(x=variable, y=aes_y, colour=interaction_col))
            + p9.geom_hline(yintercept=0, linetype="dotted")
            + geom_point(alpha=0.05)
            + theme_bw(base_size=10)
            + labs(x=x_label, y="SHAP", colour=colour_label)
            + p9.theme(legend_title=p9.element_text(angle=90, size=8), legend_title_position="right", legend_key_width=10, legend_text=p9.element_text(size=8))
        )
        if limits:
            plot += p9.xlim(limits)
        return plot

    return (shap_dependence_plot,)


@app.cell
def _(shap_dependence_plot, shap_df):
    corner_plot = shap_dependence_plot(shap_df, "dist_to_corner", limits=(-5, None))
    post_plot = shap_dependence_plot(shap_df, "dist_to_post", limits=(-5, None))
    center_plot = shap_dependence_plot(shap_df, "dist_to_center", limits=(0, 10))
    speed_plot = shap_dependence_plot(shap_df, "shot_speed")

    full_plot = (corner_plot | post_plot) / (center_plot | speed_plot)
    full_plot.save("plots/interpretation/post_shot_shap.png", width=6, height=6, dpi=500)
    full_plot
    return


if __name__ == "__main__":
    app.run()
