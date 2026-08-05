import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from polars import col as c
    from great_tables import GT, loc, style

    return GT, c, loc, pl, style


@app.cell
def _(pl):
    results = (
        pl.read_parquet("models/experiment_results_seed_48.parquet")
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
def _(pl):
    optimised_results = pl.read_parquet("models/xgboost_wce_mv_results.parquet")
    return (optimised_results,)


@app.cell
def _(optimised_results):
    optimised_results
    return


@app.cell
def _(GT, c, loc, results, style):
    (
        GT(results.drop('best_params'))
        .tab_stub(rowname_col="strategy", groupname_col="framework")
        .tab_stubhead("Imbalance Strategy")
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
    return


if __name__ == "__main__":
    app.run()
