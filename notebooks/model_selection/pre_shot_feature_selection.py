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
    import lightgbm as lgb

    from data_readers import batch_read_shot_data
    from models.features import pre_shot_features, pre_shot_features_print
    from models.data import prepare_data, DataSplitter, polars_to_pandas
    from models.training import ModelTrainer

    return (
        DataSplitter,
        ModelTrainer,
        batch_read_shot_data,
        json,
        lgb,
        polars_to_pandas,
        pre_shot_features,
        prepare_data,
        shap,
    )


@app.cell
def _(DataSplitter, batch_read_shot_data, pre_shot_features, prepare_data):
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data).collect()

    splitter = DataSplitter(data, pre_shot_features, "goal", split_path="models/train_test_20242025.npz")

    X_train, y_train, X_test, y_test, _ = splitter.get_split_data()
    return X_test, X_train, y_test, y_train


@app.cell
def _(ModelTrainer, X_test, X_train, json, y_test, y_train):
    with open("models/pre_shot/pre_shot_lightgbm_params.json", "r") as f:
        lgb_params = json.load(f)

    lgb_clf = ModelTrainer(X_train, y_train, 'lightgbm', 'WCE', loss_correct=True, seed=89, X_val=X_test, y_val=y_test, verbosity=-1, **lgb_params).train()
    return (lgb_clf,)


@app.cell
def _(lgb, lgb_clf):
    lgb.plot_metric(lgb_clf.estimator_, metric='average_precision')
    return


@app.cell
def _(X_test, lgb_clf, polars_to_pandas, shap):
    explainer = shap.TreeExplainer(lgb_clf.estimator_)
    shap_values = explainer(polars_to_pandas(X_test))
    return (shap_values,)


@app.cell
def _(shap, shap_values):
    shap.plots.beeswarm(shap_values, max_display=None)
    return


if __name__ == "__main__":
    app.run()
