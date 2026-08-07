import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, roc_auc_score
    import polars as pl
    from polars import col as c
    import shap

    from data_readers import batch_read_shot_data
    from models.features import post_shot_features_full
    from models.data import prepare_data, post_shot_filter, DataSplitter, polars_to_pandas
    from models.training import ModelTrainer

    return (
        DataSplitter,
        ModelTrainer,
        average_precision_score,
        batch_read_shot_data,
        json,
        polars_to_pandas,
        post_shot_features_full,
        post_shot_filter,
        prepare_data,
        roc_auc_score,
        shap,
    )


@app.cell
def _(
    DataSplitter,
    batch_read_shot_data,
    post_shot_features_full,
    post_shot_filter,
    prepare_data,
):
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data).pipe(post_shot_filter).collect()

    splitter = DataSplitter(data, post_shot_features_full, "goal", split_path="models/train_test_20242025.npz")

    X_train, y_train, X_test, y_test, _ = splitter.get_split_data()
    return X_test, X_train, y_test, y_train


@app.cell
def _(json):
    with open("models/post_shot/post_shot_lightgbm_params.json", "r") as f:
        params = json.load(f)
    return (params,)


@app.cell
def _(ModelTrainer, X_train, params, y_train):
    clf = ModelTrainer(X_train, y_train, 'lightgbm', 'WCE', loss_correct=True, seed=89, **params).train()
    return (clf,)


@app.cell
def _(
    X_test,
    average_precision_score,
    clf,
    polars_to_pandas,
    roc_auc_score,
    y_test,
):
    lgb_pred = clf.predict_proba(polars_to_pandas(X_test))[:,1]

    {
        'framework': 'LightGBM',
        'PR-AUC': average_precision_score(y_test, lgb_pred),
        'ROC-AUC': roc_auc_score(y_test, lgb_pred)
    }
    return


@app.cell
def _(X_test, clf, polars_to_pandas, post_shot_features_full, shap):
    explainer = shap.TreeExplainer(clf.estimator_, feature_names=post_shot_features_full)
    shap_values = explainer(polars_to_pandas(X_test))
    return (shap_values,)


@app.cell
def _(shap, shap_values):
    shap.plots.beeswarm(shap_values, max_display=None)
    return


if __name__ == "__main__":
    app.run()
