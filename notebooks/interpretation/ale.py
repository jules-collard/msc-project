# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "alibi>=0.9.0",
#     "lightgbm>=4.7.0",
#     "marimo>=0.24.0",
#     "numpy>=1.26.4",
#     "orjson>=3.12.0",
#     "plotnine[extra]>=0.15.8",
#     "polars>=1.43.2",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import joblib

    import marimo as mo
    from alibi.explainers import ALE
    from polars import col as c
    import polars as pl
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_line, theme_bw, theme, labs

    from data_readers import batch_read_shot_data
    from models.data import prepare_data, polars_to_pandas, post_shot_filter
    from models.features import pre_shot_features_minimal, post_shot_features_minimal


@app.cell
def _():
    clf = joblib.load("models/pre_shot_minimal/pre_shot_minimal_model_2008_s3870.joblib")
    clf_post = joblib.load("models/post_shot_minimal/post_shot_minimal_model_2108_s219.joblib")
    return


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026-clean/*.parquet").pipe(prepare_data).collect()
    return (shot_data,)


@app.cell
def _(shot_data):
    X = (
        shot_data
        .select(pre_shot_features_minimal)
        .with_columns(c('shot_type').cast(pl.Int16))
        .sample(50000)
        .to_numpy()
    )

    X_post = (
        shot_data
        .pipe(post_shot_filter)
        .select(post_shot_features_minimal)
        .with_columns(c('shot_type').cast(pl.Int16))
        .sample(25000)
        .to_numpy()
    )
    return


@app.cell
def _():
    # ale = ALE(clf.predict_proba, feature_names=pre_shot_features_minimal)
    # exp = ale.explain(X, min_bin_points=100)
    # joblib.dump(exp, "models/pre_shot_minimal/ale_values_20252026.joblib")
    exp = joblib.load("models/pre_shot_minimal/ale_values_20252026.joblib")
    return (exp,)


@app.cell
def _():
    # ale_post = ALE(clf_post.predict_proba, feature_names=post_shot_features_minimal)
    # exp_post = ale_post.explain(X_post, min_bin_points=100)
    # joblib.dump(exp_post, "models/post_shot_minimal/ale_values_20252026.joblib")
    exp_post = joblib.load("models/post_shot_minimal/ale_values_20252026.joblib")
    return


@app.cell
def _():
    pre_shot_features_minimal
    return


@app.function
def plot_ale(explanation, feature_name, smooth=True):
    feature_index = pre_shot_features_minimal.index(feature_name)

    df = pl.DataFrame({
        'ALE': explanation.ale_values[feature_index][:,1],
        feature_name: explanation.feature_values[feature_index]
    })

    plot = (
        ggplot(df, aes(x=feature_name, y="ALE"))
        + p9.geom_hline(yintercept=0, linetype="dashed")
        + geom_point(size=0.5, alpha=0.5)
        + theme_bw(base_size=10)
    )

    if smooth:
        plot += p9.geom_smooth(colour="red", se=False)
    else:
        plot += p9.geom_line()
    return plot


@app.cell
def _(exp):
    plot_ale(exp, "visible_angle", smooth=True)
    return


if __name__ == "__main__":
    app.run()
