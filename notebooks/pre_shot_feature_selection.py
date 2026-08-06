import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    from sklearn.calibration import calibration_curve
    import polars as pl
    from polars import col as c
    import shap

    from data_readers import batch_read_shot_data
    from models.features import pre_shot_features, pre_shot_features_print
    from models.data import prepare_data, DataSplitter
    from models.training import ModelTrainer

    return (
        DataSplitter,
        ModelTrainer,
        batch_read_shot_data,
        json,
        pre_shot_features,
        pre_shot_features_print,
        prepare_data,
        shap,
    )


@app.cell
def _(DataSplitter, batch_read_shot_data, pre_shot_features, prepare_data):
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data).collect()

    splitter = DataSplitter(data, pre_shot_features, "goal", split_path="models/train_test_20242025.npz")

    X_train, y_train, X_test, y_test, _ = splitter.get_split_data()
    return X_test, X_train, y_train


@app.cell
def _(json):
    with open("models/pre_shot/pre_shot_xgboost_params.json", "r") as f:
        params = json.load(f)
    return (params,)


@app.cell
def _(ModelTrainer, X_train, params, y_train):
    clf = ModelTrainer(X_train, y_train, 'xgboost', 'WCE', loss_correct=True, seed=89, **params).train()
    return (clf,)


@app.cell
def _(X_test, clf, pre_shot_features_print, shap):
    explainer = shap.TreeExplainer(clf.estimator_, feature_names=pre_shot_features_print)
    shap_values = explainer(X_test)
    return (shap_values,)


@app.cell
def _(shap, shap_values):
    shap.plots.beeswarm(shap_values, max_display=None)
    return


if __name__ == "__main__":
    app.run()
