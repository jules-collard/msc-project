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
    from models.features import pre_shot_features, pre_shot_features_print
    from models.data import prepare_data, DataSplitter, polars_to_pandas
    from models.training import ModelTrainer

    return (
        DataSplitter,
        ModelTrainer,
        average_precision_score,
        batch_read_shot_data,
        json,
        polars_to_pandas,
        pre_shot_features,
        pre_shot_features_print,
        prepare_data,
        roc_auc_score,
        shap,
    )


@app.cell
def _(DataSplitter, batch_read_shot_data, pre_shot_features, prepare_data):
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data).collect()

    splitter = DataSplitter(data, pre_shot_features, "goal", split_path="models/train_test_20242025.npz")

    X_train, y_train, X_test, y_test, _ = splitter.get_split_data()
    return X_test, X_train, y_test, y_train


@app.cell
def _(json):
    with open("models/pre_shot/pre_shot_lightgbm_params.json", "r") as f:
        lgb_params = json.load(f)

    with open("models/pre_shot/pre_shot_xgboost_params.json", "r") as f:
        xgb_params = json.load(f)
    return lgb_params, xgb_params


@app.cell
def _(ModelTrainer, X_train, lgb_params, xgb_params, y_train):
    lgb_clf = ModelTrainer(X_train, y_train, 'lightgbm', 'WCE', loss_correct=True, seed=89, **lgb_params).train()
    xgb_clf = ModelTrainer(X_train, y_train, 'xgboost', 'WCE', loss_correct=True, seed=105, **xgb_params).train()
    return lgb_clf, xgb_clf


@app.cell
def _(
    X_test,
    average_precision_score,
    lgb_clf,
    polars_to_pandas,
    roc_auc_score,
    xgb_clf,
    y_test,
):
    lgb_pred = lgb_clf.predict_proba(polars_to_pandas(X_test))[:,1]
    xgb_pred = xgb_clf.predict_proba(X_test)[:,1]

    results = [
        {
            'framework': 'XGBoost',
            'PR-AUC': average_precision_score(y_test, xgb_pred),
            'ROC-AUC': roc_auc_score(y_test, xgb_pred)
        },
            {
            'framework': 'LightGBM',
            'PR-AUC': average_precision_score(y_test, lgb_pred),
            'ROC-AUC': roc_auc_score(y_test, lgb_pred)
        }
    ]
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(X_test, clf, polars_to_pandas, pre_shot_features_print, shap):
    explainer = shap.TreeExplainer(clf.estimator_, feature_names=pre_shot_features_print)
    shap_values = explainer(polars_to_pandas(X_test))
    return (shap_values,)


@app.cell
def _(shap, shap_values):
    shap.plots.beeswarm(shap_values, max_display=None)
    return


if __name__ == "__main__":
    app.run()
