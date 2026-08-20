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
    from models.features import pre_shot_features_minimal, post_shot_features_full, pre_shot_features_print


@app.cell
def _():
    shot_data = batch_read_shot_data("/scratch/shot_data/20252026/*.parquet")
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
def _():
    # X = data.select(pre_shot_features_minimal)
    # explainer = shap.TreeExplainer(pre_shot_model.estimator_)
    # shap_values = explainer(polars_to_pandas(X))
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


@app.function
def shap_dependence_plot(df: pl.DataFrame, variable: str, variable_title = None, smooth=True):
    aes_y = f"{variable}_shap"
    x_label = variable_title if variable_title else variable
    plot = (
        ggplot(df, aes(x=variable, y=aes_y))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + geom_point(alpha=0.05)
        + theme_bw(base_size=10)
        + labs(x=x_label, y="SHAP Value (log-odds)")
    )
    if smooth:
        plot = plot + geom_smooth(data=df.sample(999), colour="red", se=False)
    return plot


@app.cell
def _(shap_df):
    distance_plot = shap_dependence_plot(shap_df, "shooter_dist_to_goal", "Shooter Distance to Goal (ft)")
    angle_plot = shap_dependence_plot(shap_df, "shooter_angle_to_goal", "Shooter Angle to Goal (degrees)")
    visible_angle_plot = shap_dependence_plot(shap_df, "visible_angle", "Visible Angle (degrees)")
    goal_speed_plot = shap_dependence_plot(shap_df, "shooter_goal_speed", "Shooter Goalwards Speed (ft/s)")

    (distance_plot | angle_plot) / (visible_angle_plot | goal_speed_plot)
    return


@app.cell
def _():
    pre_shot_features_minimal
    return


@app.cell
def _(shap_df):
    goalie_distance_plot = shap_dependence_plot(shap_df, "goalie_dist_to_goal", "Goalie Distance to Goal (ft)", smooth=False)
    goalie_lateral_speed_plot = shap_dependence_plot(shap_df, "goalie_lateral_speed", "Goalie Lateral Speed (ft/s)")
    goalie_angle_plot = shap_dependence_plot(shap_df, "goalie_angle_to_shooter", "Goalie Angle to Shooter (degrees)", smooth=False)
    pressure_plot = shap_dependence_plot(shap_df, "total_pressure", "Defensive Pressure")

    (goalie_angle_plot) / (goalie_lateral_speed_plot | pressure_plot)
    return


@app.cell
def _(shap_df):
    plot_data = shap_df.with_columns(c('num_defenders_in_shadow_lane', 'goalie_in_shooting_lane').cast(pl.String))

    defender_lane_plot = (
        ggplot(plot_data, aes(x="num_defenders_in_shadow_lane", y="num_defenders_in_shadow_lane_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1, show_legend=False)
        + theme_bw(base_size=10)
        + labs(x="Defenders in Shadow Lane", y="SHAP Value (log-odds)", colour="Distance to Goal")
    )

    goalie_lane_plot = (
        ggplot(plot_data, aes(x="goalie_in_shooting_lane", y="goalie_in_shooting_lane_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1, show_legend=False)
        + theme_bw(base_size=10)
        + labs(x="Goalie in Shadow Lane", y="SHAP Value (log-odds)")
    )

    shot_type_plot = (
        ggplot(plot_data, aes(x="shot_type", y="shot_type_shap", colour="shooter_dist_to_goal"))
        + p9.geom_hline(yintercept=0, linetype="dotted")
        + p9.geom_sina(alpha=0.1)
        + theme_bw(base_size=10)
        + labs(x="Shot Type", y="SHAP Value (log-odds)", colour="Distance to Goal")
        + p9.theme(axis_text_x=p9.element_text(rotation=45))
    )

    (defender_lane_plot | goalie_lane_plot) / shot_type_plot
    return


if __name__ == "__main__":
    app.run()
