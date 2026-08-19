import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json

    import marimo as mo
    import polars as pl
    from polars import col as c
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.calibration import calibration_curve
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_line, labs, theme_bw
    from great_tables import GT, loc, style
    from matplotlib import pyplot as plt
    import shap
    from calibra.errors import classwise_ece

    from data_readers import batch_read_shot_data
    from models.data import DataSplitter, prepare_data, polars_to_pandas
    from models.training import ModelTrainer
    from models.features import pre_shot_features, pre_shot_features_pruned, pre_shot_features_minimal, pre_shot_features_print


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Model Framework & Imbalance Strategy Comparison
    """)
    return


@app.cell
def _():
    results = (
        pl.read_parquet("models/pre_shot/fixed_trees_experiment_results_s3557.parquet")
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
            title="Pre-Shot Model Comparison",
        ).tab_source_note(
            "Scores shown are cross-validated means on training set."
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
    Conduct more thorough hyperparameter search on LightGBM + WCE with:
    ```
    PYTHONPATH=scripts/ uv run python -m tools.tune_hyperparameters pre_shot_pruned --data-pattern "/output/shot_data/20242025/*.parquet" --split-path "models/train_test_20242025.npz" --params-file "models/pre_shot/pre_shot_lightgbm_params_s6906.json" --seed 6906 --frameworks lightgbm --strategies WCE --multivariate --n-trials 200 --info-log --n-estimators 1500
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Selection
    """)
    return


@app.cell
def _():
    data = (
        batch_read_shot_data("/output/shot_data/20242025/*.parquet")
        .pipe(prepare_data)
        .collect()
    )

    splitter = DataSplitter(data, pre_shot_features, "goal", split_path="models/train_test_20242025.npz")

    X_train, y_train, X_test, y_test, _ = splitter.get_split_data()
    return X_test, X_train, data, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    with open("models/pre_shot/pre_shot_full_lightgbm_params_s3590.json", "r") as f:
        params = json.load(f)

    clf = ModelTrainer(X_train, y_train, 'lightgbm', 'WCE', loss_correct=True, seed=104, X_val=X_test, y_val=y_test, **params).train()
    return (clf,)


@app.cell(disabled=True)
def _(X_test, clf):
    explainer = shap.TreeExplainer(clf.estimator_)

    X_test_pd = polars_to_pandas(X_test)
    X_test_pd.columns = pre_shot_features_print
    shap_values = explainer(X_test_pd)
    return (shap_values,)


@app.cell(disabled=True)
def _(shap_values):
    shap.plots.beeswarm(shap_values, max_display=None, show=False)
    plt.gca()
    plt.savefig("plots/model_selection/pre_shot_feature_importance.svg", bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We drop `num_pressures_right` and `num_pressures_left`, getting optimal hyperparameters with
    ```
    PYTHONPATH=scripts/ uv run python -m tools.tune_hyperparameters pre_shot --data-pattern "/output/shot_data/20242025/*.parquet" --split-path "models/train_test_20242025.npz" --params-file "models/pre_shot/pre_shot_full_lightgbm_params_s3590.json" --seed 3590 --frameworks lightgbm --strategies WCE --multivariate --n-trials 200 --info-log --n-estimators 1500
    ```
    """)
    return


@app.cell
def _():
    with open("models/pre_shot/pre_shot_lightgbm_params.json", "r") as f_pruned:
        params_pruned = json.load(f_pruned)

    with open("models/pre_shot_minimal/pre_shot_minimal_lightgbm_params_s2487.json", "r") as f_minimal:
        params_minimal = json.load(f_minimal)
    return params_minimal, params_pruned


@app.cell
def _(data, params_minimal, params_pruned):
    splitter_pruned = DataSplitter(data, pre_shot_features_pruned, "goal", split_path="models/train_test_20242025.npz")
    splitter_minimal = DataSplitter(data, pre_shot_features_minimal, "goal", split_path="models/train_test_20242025.npz")

    X_train_pruned, y_train_pruned, X_test_pruned, y_test_pruned, _pruned = splitter_pruned.get_split_data()
    X_train_minimal, y_train_minimal, X_test_minimal, y_test_minimal, _minimal = splitter_minimal.get_split_data()

    clf_pruned = ModelTrainer(X_train_pruned, y_train_pruned, 'lightgbm', 'WCE', loss_correct=True, seed=104, X_val=X_test_pruned, y_val=y_test_pruned, **params_pruned).train()
    clf_minimal = ModelTrainer(X_train_minimal, y_train_minimal, 'lightgbm', 'WCE', loss_correct=True, seed=104, X_val=X_test_minimal, y_val=y_test_pruned, **params_minimal).train()
    return (
        X_test_minimal,
        X_test_pruned,
        clf_minimal,
        clf_pruned,
        y_test_minimal,
        y_test_pruned,
    )


@app.cell
def _(
    X_test,
    X_test_minimal,
    X_test_pruned,
    clf,
    clf_minimal,
    clf_pruned,
    y_test,
    y_test_minimal,
    y_test_pruned,
):
    pred_full = clf.predict_proba(polars_to_pandas(X_test))[:,1]
    pred_pruned = clf_pruned.predict_proba(polars_to_pandas(X_test_pruned))[:,1]
    pred_minimal = clf_minimal.predict_proba(polars_to_pandas(X_test_minimal))[:,1]

    pruning_results = pl.from_dicts([
        {
            'Feature Set': 'Full',
            'PR-AUC': average_precision_score(y_test, pred_full),
            'ROC-AUC': roc_auc_score(y_test, pred_full)
        },
        {
            'Feature Set': 'Pruned',
            'PR-AUC': average_precision_score(y_test_pruned, pred_pruned),
            'ROC-AUC': roc_auc_score(y_test_pruned, pred_pruned)
        },
        {
            'Feature Set': 'Minimal',
            'PR-AUC': average_precision_score(y_test_minimal, pred_minimal),
            'ROC-AUC': roc_auc_score(y_test_minimal, pred_minimal)
        },
    ])
    return pred_minimal, pred_pruned, pruning_results


@app.cell
def _(pruning_results):
    pruning_table = (
        GT(pruning_results)
        .tab_header(
            title="Pre-Shot Feature Set Comparison",
        ).tab_source_note(
            "Scores shown are calculated on validation set."
        )
        .fmt_number(columns=["PR-AUC", "ROC-AUC"], decimals=4)
    )

    pruning_table
    # print(pruning_table.as_latex())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Calibration
    """)
    return


@app.cell
def _(pred_minimal, y_test_minimal):
    prob_true, prob_pred = calibration_curve(y_test_minimal, pred_minimal, n_bins=6)

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
def _(pred_pruned, y_test_pruned):
    classwise_ece(pred_pruned, y_test_pruned, num_bins=6)
    return


if __name__ == "__main__":
    app.run()
