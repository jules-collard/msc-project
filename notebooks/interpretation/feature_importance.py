import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import joblib

    import shap
    import matplotlib.pyplot as plt

    from data_readers import batch_read_shot_data
    from models.data import prepare_data, post_shot_filter, polars_to_pandas
    from models.features import post_shot_features_full, pre_shot_features_pruned


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026/*.parquet").pipe(prepare_data).collect()
    pre_shot_model = joblib.load("models/pre_shot/pre_shot_model_2008_s2339.joblib")
    post_shot_model = joblib.load("models/post_shot/post_shot_model_2108_s3475.joblib")
    return post_shot_model, shot_data


@app.cell
def _():
    # X = polars_to_pandas(shot_data.select(pre_shot_features_pruned))
    # explainer = shap.TreeExplainer(pre_shot_model.estimator_)
    # shap_values = explainer(X)
    # joblib.dump(shap_values, "models/pre_shot/shap_values_20252026.joblib")
    shap_values = joblib.load("models/pre_shot/shap_values_20252026.joblib")
    return (shap_values,)


@app.cell
def _(shap_values):
    shap.plots.beeswarm(shap_values, max_display=None, show=False)
    plt.gca()
    # plt.savefig("plots/interpretation/pre_shot_importance.png", dpi=500, bbox_inches='tight')
    return


@app.cell
def _(post_shot_model, shot_data):
    X_post = polars_to_pandas(shot_data.pipe(post_shot_filter).select(post_shot_features_full))
    explainer = shap.TreeExplainer(post_shot_model.estimator_)
    shap_values_post = explainer(X_post)
    joblib.dump(shap_values_post, "models/post_shot/shap_values_20252026.joblib")
    # shap_values_post = joblib.load("models/post_shot/shap_values_20252026.joblib")
    return (shap_values_post,)


@app.cell
def _(shap_values_post):
    shap.plots.beeswarm(shap_values_post, max_display=None, show=False)
    plt.gca()
    plt.savefig("plots/interpretation/post_shot_importance.png", dpi=500, bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
