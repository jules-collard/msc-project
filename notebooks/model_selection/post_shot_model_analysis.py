import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import json

    import marimo as mo
    from great_tables import GT, loc, style
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, roc_auc_score
    import polars as pl
    from polars import col as c
    import shap
    from matplotlib import pyplot as plt
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_line, labs, theme_bw
    from calibra.errors import classwise_ece

    from data_readers import batch_read_shot_data
    from models.features import post_shot_features_full, post_shot_features_minimal
    from models.data import prepare_data, post_shot_filter, DataSplitter, polars_to_pandas
    from models.training import ModelTrainer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Model Framework & Imbalance Strategy Comparison
    """)
    return


@app.cell
def _():
    results = (
        pl.read_parquet("models/post_shot/fixed_trees_experiment_results_s7146.parquet")
        .with_columns(
            pl.col('framework')
            .str.replace('xgboost', 'XGBoost', literal=True)
            .str.replace('lightgbm', 'LightGBM', literal=True)
            .str.replace('dart', 'DART', literal=True)
        ).with_columns(
            pl.col('strategy')
            .str.replace('RO', 'Oversampling')
            .str.replace('RU', 'Undersampling')
            .str.replace('WCE', 'WCE Loss')
        ).sort(pl.col('framework', 'strategy'))
    )
    return (results,)


@app.cell
def _(results):
    experiments_table = (
        GT(results.drop('best_params'))
        .tab_header(
            title="Post-Shot Model Comparison",
        ).tab_source_note(
            "Scores are cross-validated means on training set."
        )
        .tab_stub(rowname_col="strategy", groupname_col="framework")
        .tab_stubhead("Model")
        .cols_label(
            best_pr_auc="PR-AUC",
            best_roc_auc="ROC-AUC"
        )
        .fmt_number(columns=["best_pr_auc", "best_roc_auc"], decimals=4)
        .tab_style(
            style.text(decorate='underline', style='oblique'),
            loc.body(
                columns=c('best_pr_auc'),
                rows=(c('best_pr_auc') == c('best_pr_auc').max()).over('framework')
            )
        ).tab_style(
            style.text(decorate='underline', style='oblique'),
            loc.body(
                columns=c('best_roc_auc'),
                rows=(c('best_roc_auc') == c('best_roc_auc').max()).over('framework')
            )
        ).tab_style(
            style.text(weight='bolder'),
            loc.body(
                columns=c('best_pr_auc'),
                rows=(c('best_pr_auc') == c('best_pr_auc').max())
            )
        ).tab_style(
            style.text(weight='bold'),
            [loc.column_labels(), loc.stubhead()]
        )
        .tab_options(
            row_group_as_column=True,
            table_body_vlines_style="None",
            table_body_hlines_style="None",
            stub_row_group_border_style="",
        )
    )

    experiments_table
    # print(experiments_table.as_latex())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Set Comparison
    """)
    return


@app.cell
def _():
    xg_data = pl.scan_parquet("/output/predictions/20242025/pre_shot_1308.parquet")
    data = (
        batch_read_shot_data("/output/shot_data/20242025/*.parquet")
        .pipe(prepare_data)
        .pipe(post_shot_filter)
        .join(xg_data, on=["game_id", "period", "shot_id"], how='left', validate='1:1')
        .collect()
    )

    splitter_full = DataSplitter(data, post_shot_features_full, "goal", split_path="models/train_test_20242025.npz")
    splitter_min = DataSplitter(data, post_shot_features_minimal, "goal", split_path="models/train_test_20242025.npz")

    X_train_full, y_train_full, X_test_full, y_test_full, _ = splitter_full.get_split_data()
    X_train_min, y_train_min, X_test_min, y_test_min, _ = splitter_min.get_split_data()
    return (
        X_test_full,
        X_test_min,
        X_train_full,
        X_train_min,
        y_test_full,
        y_test_min,
        y_train_full,
        y_train_min,
    )


@app.cell
def _():
    with open("models/post_shot/post_shot_lightgbm_params.json", "r") as f:
        params_full = json.load(f)

    with open("models/post_shot_minimal/post_shot_minimal_lightgbm_params_s7160.json", "r") as f:
        params_min = json.load(f)
    return params_full, params_min


@app.cell
def _(
    X_test_full,
    X_test_min,
    X_train_full,
    X_train_min,
    params_full,
    params_min,
    y_test_full,
    y_test_min,
    y_train_full,
    y_train_min,
):
    clf_full = ModelTrainer(X_train_full, y_train_full, 'lightgbm', 'WCE', loss_correct=True, seed=89, X_val=X_test_full, y_val=y_test_full, **params_full).train()

    clf_min = ModelTrainer(X_train_min, y_train_min, 'lightgbm', 'WCE', loss_correct=True, seed=89, X_val=X_test_min, y_val=y_test_min, **params_min).train()
    return clf_full, clf_min


@app.cell
def _(X_test_full, X_test_min, clf_full, clf_min, y_test_full, y_test_min):
    pred_full = clf_full.predict_proba(polars_to_pandas(X_test_full))[:,1]
    pred_min = clf_min.predict_proba(polars_to_pandas(X_test_min))[:,1]

    feature_results = pl.from_dicts([
        {
            'Feature Set': 'Pre-Shot + Post-Shot',
            'PR-AUC': average_precision_score(y_test_full, pred_full),
            'ROC-AUC': roc_auc_score(y_test_full, pred_full)
        },
        {
            'Feature Set': 'PreXG + Post-Shot',
            'PR-AUC': average_precision_score(y_test_min, pred_min),
            'ROC-AUC': roc_auc_score(y_test_min, pred_min)
        },
    ])
    return feature_results, pred_full


@app.cell
def _(feature_results):
    features_table = (
        GT(feature_results)
        .tab_header(
            title="Post-Shot Feature Set Comparison"
        ).tab_source_note(
            "Scores calculated on validation set."
        )
        .fmt_number(columns=["PR-AUC", "ROC-AUC"], decimals=4)
    )

    features_table
    # print(features_table.as_latex())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Importance Analysis
    """)
    return


@app.cell(disabled=True)
def _(X_test_full, clf_full):
    explainer = shap.TreeExplainer(clf_full.estimator_, feature_names=post_shot_features_full)
    shap_values = explainer(polars_to_pandas(X_test_full))
    return (shap_values,)


@app.cell(disabled=True)
def _(shap_values):
    shap.plots.beeswarm(shap_values, max_display=None, show=False)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Calibration
    """)
    return


@app.cell
def _(pred_full, y_test_full):
    prob_true, prob_pred = calibration_curve(y_test_full, pred_full, n_bins=6)

    (
        pl.DataFrame({
            'mean_predicted_probability': prob_pred,
            'fraction_of_positives': prob_true
        })
        >> ggplot(aes(x="mean_predicted_probability", y="fraction_of_positives"))
        + p9.geom_abline(slope=1, intercept=0, linetype="dashed", alpha=0.6)
        + geom_line()
        + geom_point()
        + p9.xlim(0,1)
        + p9.ylim(0,1)
        + labs(x="Mean Predicted Probability", y="Fraction of Positives")
        + theme_bw(base_size=10)
    )
    return


@app.cell
def _(pred_full, y_test_full):
    classwise_ece(pred_full, y_test_full, num_bins=6)
    return


if __name__ == "__main__":
    app.run()
