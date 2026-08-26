import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import joblib

    import marimo as mo
    import numpy as np
    import polars as pl
    from polars import selectors as cs
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_col, theme_bw, labs
    import shap

    from data_readers import batch_read_shot_data
    from models.data import prepare_data, post_shot_filter, polars_to_pandas
    from models.features import post_shot_features_full


@app.cell
def _():
    model = joblib.load("models/post_shot/post_shot_model_2108_s3475.joblib")
    return


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026/*.parquet")
    predictions = pl.scan_parquet("/output/predictions/20252026/post_shot_2108.parquet")
    return predictions, shot_data


@app.cell
def _(predictions, shot_data):
    data = (
        shot_data
        .pipe(prepare_data)
        .pipe(post_shot_filter)
        .join(
            predictions,
            on=["game_id", "period", "shot_id"],
            how="left",
            validate="1:1"
        )
        .collect()
    )
    return (data,)


@app.cell
def _(data):
    X = polars_to_pandas(data.select(post_shot_features_full))
    # explainer = shap.TreeExplainer(model.estimator_)
    # shap_values = explainer(X)
    # joblib.dump(shap_values, "models/post_shot/shap_values_20252026.joblib")
    shap_values = joblib.load("models/post_shot/shap_values_20252026.joblib")
    return (shap_values,)


@app.cell
def _(data, shap_values):
    shap_values_df = pl.DataFrame(shap_values.values, schema=[f"{feature}_shap" for feature in post_shot_features_full])
    base_values_df = pl.DataFrame(shap_values.base_values, schema=["base_value"])

    full_df = pl.concat([data, shap_values_df, base_values_df], how="horizontal", strict=True)
    return (full_df,)


@app.cell
def _():
    shot_origin = [
        "shot_x",
        "shot_y",
        "shooter_dist_to_goal",
        "shooter_angle_to_goal",
        "visible_angle",
    ]

    shot_type = [
        "shot_type"
    ]

    shooter_movement = [
        "shooter_speed",
        "shooter_lateral_speed",
        "shooter_goal_speed",
    ]

    defender_features = [
        "total_pressure",
        "num_defenders_in_shooting_lane",
        "num_defenders_in_shadow_lane",
        "num_pressures_front",
        "num_pressures_back",
    ]

    goalie_features = [
        "goalie_angle_to_shooter",
        "goalie_in_shooting_lane",
        "goalie_in_shadow_lane",
        "goalie_dist_to_goal",
        "goalie_speed",
        "goalie_lateral_speed",
    ]

    shot_execution = [
        "shot_speed",
        "goalline_y_norm",
        "goalline_z",
        "on_goal",
        "dist_to_post",
        "dist_to_corner",
        "dist_to_center"
    ]
    return (
        defender_features,
        goalie_features,
        shooter_movement,
        shot_execution,
        shot_origin,
        shot_type,
    )


@app.cell
def _(
    defender_features,
    full_df,
    goalie_features,
    shooter_movement,
    shot_execution,
    shot_origin,
    shot_type,
):
    explanations = (
        full_df
        .select(c('game_id', 'period', 'shot_id', 'player_reference_id', 'player_first_name', 'player_last_name', 'sportlogiq_xg', 'post_shot', 'base_value'), cs.ends_with('shap'))
        .select(pl.all().name.replace('_shap', '', literal=True))
        .with_columns(
            pl.sum_horizontal(shot_type).alias('Shot Type'),
            pl.sum_horizontal(shot_execution).alias('Shot Execution'),
            pl.sum_horizontal(shot_origin).alias('Shot Origin'),
            pl.sum_horizontal(shooter_movement).alias('Shooter Movement'),
            pl.sum_horizontal(defender_features).alias('Defender Positioning'),
            pl.sum_horizontal(goalie_features).alias('Goalie Positioning'),
        ).unpivot(
            on=["Shot Origin", "Shooter Movement", "Shot Type", "Defender Positioning", "Goalie Positioning", "Shot Execution"],
            index=['game_id', 'period', 'shot_id', 'player_reference_id', 'player_first_name', 'player_last_name', 'sportlogiq_xg', 'post_shot', 'base_value'],
            variable_name="source",
            value_name="shap"
        ).with_columns(
            increase=c('shap') > 0,
            abs_shap=c('shap').abs(),
            label_coord = pl.when(c('shap') > 0).then(c('shap') + 0.15).otherwise(c('shap') - 0.15),
            diff=c('post_shot') - c('sportlogiq_xg')
        )
    )
    return (explanations,)


@app.cell
def _(explanations):
    game_ids = explanations.select(c('game_id').unique()).to_series()
    game_id_selector = mo.ui.dropdown(game_ids, value=game_ids.first())
    return (game_id_selector,)


@app.cell
def _(explanations, game_id_selector):
    shot_ids = explanations.filter(c('game_id') == game_id_selector.value).select(c('shot_id').unique()).to_series()
    shot_id_selector = mo.ui.dropdown(shot_ids, value=shot_ids.first())
    return (shot_id_selector,)


@app.cell
def _(explanations, game_id_selector, shot_id_selector):
    local_explanation_plot = (
        explanations
        .filter(
            c('game_id') == game_id_selector.value,
            c('shot_id') == shot_id_selector.value
        )
        >> ggplot(aes(x="reorder(source, abs_shap)", y="shap"))
        + geom_col(aes(fill="increase"), show_legend=False)
        + p9.geom_text(aes(label="shap", y="label_coord"), format_string="{:+.2f}")
        + theme_bw(base_size=12)
        + p9.coord_flip()
        + labs(y="SHAP (log-odds)", x="")
    )
    return (local_explanation_plot,)


@app.cell
def _(game_id_selector, local_explanation_plot, shot_id_selector):
    mo.vstack([
        mo.hstack([game_id_selector, shot_id_selector], justify="start"),
        local_explanation_plot
    ])
    return


@app.cell
def _():
    # local_explanation_plot.save("plots/interpretation/local_example.svg")
    return


if __name__ == "__main__":
    app.run()
