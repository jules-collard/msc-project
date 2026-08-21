import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from great_tables import GT
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_line, geom_text, theme_bw, labs
    import polars as pl
    from polars import col as c
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
    from sklearn.calibration import calibration_curve
    from calibra.errors import classwise_ece

    from data_readers import batch_read_shot_data
    from models.data import prepare_data, post_shot_filter


@app.cell
def _():
    shot_data = batch_read_shot_data("/output/shot_data/20252026-clean/*.parquet")
    pre_shot_xg = pl.scan_parquet("/output/predictions/20252026/pre_shot_2008.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/20252026/post_shot_2108.parquet")
    return post_shot_xg, pre_shot_xg, shot_data


@app.cell
def _(post_shot_xg, pre_shot_xg, shot_data):
    data = (
        shot_data
        .pipe(prepare_data)
        .join(
            pre_shot_xg,
            on=["game_id", "period", "shot_id"],
            how='left',
            validate='1:1'
        ).join(
            post_shot_xg,
            on=["game_id", "period", "shot_id"],
            how='left',
            validate='1:1'
        ).with_columns(
            pl.when(c('type').str.contains('blocked')).then(0).otherwise(c('post_shot')).alias('post_shot'),
        ).with_columns(
            c('post_shot').fill_null(c('pre_shot')).alias('post_shot_imputed')
        )
        # .select(c('game_id', 'period', 'shot_id', 'goal', 'sportlogiq_xg', 'pre_shot', 'post_shot'))
        .collect()
    )
    return (data,)


@app.function
def get_predictions(data: pl.DataFrame):
    goals = data.select(c('goal')).to_series().to_list()
    sportlogiq = data.select(c('sportlogiq_xg')).to_series().to_list()
    pre_shot = data.select(c('pre_shot')).to_series().to_list()
    post_shot = data.select(c('post_shot')).to_series().to_list()
    post_shot_imputed = data.select(c('post_shot_imputed')).to_series().to_list()

    return goals, sportlogiq, pre_shot, post_shot, post_shot_imputed


@app.cell
def _(data):
    goals, sportlogiq, pre_shot, _, post_shot_imputed = get_predictions(data)

    post_shot_valid = data.filter(c('post_shot').is_not_null())
    post_shot = post_shot_valid.select(c('post_shot')).to_series().to_list()
    goals_post_shot = post_shot_valid.select(c('goal')).to_series().to_list()


    model_metrics = pl.DataFrame({
        'model': ['Sportlogiq', 'Pre-Shot', 'Post-Shot', 'Post-Shot (Imputed)'],
        'prauc': [
            average_precision_score(goals, sportlogiq),
            average_precision_score(goals, pre_shot),
            average_precision_score(goals_post_shot, post_shot),
            average_precision_score(goals, post_shot_imputed),
        ],
        'rocauc': [
            roc_auc_score(goals, sportlogiq),
            roc_auc_score(goals, pre_shot),
            roc_auc_score(goals_post_shot, post_shot),
            roc_auc_score(goals, post_shot_imputed),
        ],
        'ece': [
            classwise_ece(sportlogiq, goals, method='frequency'),
            classwise_ece(pre_shot, goals, method='frequency'),
            classwise_ece(post_shot, goals_post_shot, method='frequency'),
            classwise_ece(post_shot_imputed, goals, method='frequency'),
        ]
    })
    return (
        goals,
        goals_post_shot,
        model_metrics,
        post_shot,
        pre_shot,
        sportlogiq,
    )


@app.cell
def _(model_metrics):
    metrics_table = (
        GT(model_metrics)
        .tab_header(title="Model Evaluation Metrics")
        .tab_source_note("Scores are evaluated on test set (2025-26 Season)")
        .tab_source_note("Post-Shot models set blocked shots to 0")
        .cols_label(model="Model", prauc="PR-AUC", rocauc="ROC-AUC", ece="ECE")
        .tab_stub(rowname_col='model')
        .sub_missing()
        .fmt_number(['prauc', 'rocauc', 'ece'], decimals=4)
    )

    metrics_table
    # print(metrics_table.as_latex())
    return


@app.cell
def _(goals, goals_post_shot, post_shot, pre_shot, sportlogiq):
    pre_precision, pre_recall, _ = precision_recall_curve(goals, pre_shot, drop_intermediate=True)
    post_precision, post_recall, _ = precision_recall_curve(goals_post_shot, post_shot, drop_intermediate=True)
    sportlogiq_precision, sportlogiq_recall, _ = precision_recall_curve(goals, sportlogiq, drop_intermediate=True)
    return (
        post_precision,
        post_recall,
        pre_precision,
        pre_recall,
        sportlogiq_precision,
        sportlogiq_recall,
    )


@app.cell
def _(goals, goals_post_shot, post_shot, pre_shot, sportlogiq):
    pre_fpr, pre_tpr, __ = roc_curve(goals, pre_shot, drop_intermediate=True)
    post_fpr, post_tpr, __ = roc_curve(goals_post_shot, post_shot, drop_intermediate=True)
    sportlogiq_fpr, sportlogiq_tpr, __ = roc_curve(goals, sportlogiq, drop_intermediate=True)
    return post_fpr, post_tpr, pre_fpr, pre_tpr, sportlogiq_fpr, sportlogiq_tpr


@app.cell
def _(data):
    base_rate = data.select(c('goal')).mean().item()
    y_breaks = [0.00, base_rate, 0.25, 0.50, 0.75, 1.00]

    # 2. Define the exact text for each corresponding tick
    y_labels = ["0.00", f"{base_rate:.3f}", "0.25", "0.50", "0.75", "1.00"]
    return base_rate, y_breaks, y_labels


@app.cell
def _(
    base_rate,
    post_precision,
    post_recall,
    pre_precision,
    pre_recall,
    sportlogiq_precision,
    sportlogiq_recall,
    y_breaks,
    y_labels,
):
    df_pre = pl.DataFrame({
        'model': 'Pre-Shot',
        'precision': pre_precision,
        'recall': pre_recall,
    })

    df_sportlogiq = pl.DataFrame({
        'model': 'Sportlogiq',
        'precision': sportlogiq_precision,
        'recall': sportlogiq_recall,
    })

    df_post = pl.DataFrame({
        'model': 'Post-Shot',
        'precision': post_precision,
        'recall': post_recall,
    })

    pr_df = pl.concat([df_pre, df_sportlogiq, df_post])

    pr_plot = (
        ggplot(pr_df, aes(x='recall', y='precision', color='model'))
        + geom_line()
        + p9.geom_hline(yintercept=base_rate, linetype="dashed")
        + theme_bw(base_size=12)
        + p9.xlim(0,1)
        + labs(y="Precison", x="Recall", color="Model", linetype="Data")
        + p9.scale_y_continuous(labels=y_labels, breaks=y_breaks)
    )

    # pr_plot.save("plots/evaluation/pr_curve.svg")
    pr_plot
    return


@app.cell
def _(post_fpr, post_tpr, pre_fpr, pre_tpr, sportlogiq_fpr, sportlogiq_tpr):
    df_roc_pre = pl.DataFrame({
        'model': 'Pre-Shot',
        'fpr': pre_fpr,
        'tpr': pre_tpr,
    })

    df_roc_sportlogiq = pl.DataFrame({
        'model': 'Sportlogiq',
        'fpr': sportlogiq_fpr,
        'tpr': sportlogiq_tpr,
    })

    df_roc_post = pl.DataFrame({
        'model': 'Post-Shot',
        'fpr': post_fpr,
        'tpr': post_tpr,
    })

    roc_df = pl.concat([df_roc_pre, df_roc_sportlogiq, df_roc_post])

    roc_plot = (
        ggplot(roc_df, aes(x='fpr', y='tpr', color='model'))
        + geom_line()
        + p9.geom_abline(linetype='dashed')
        + theme_bw(base_size=12)
        + p9.xlim(0,1)
        + p9.ylim(0,1)
        + labs(y="True Positive Rate", x="False Positive Rate", color="Model", linetype="Data")
    )

    # roc_plot.save("plots/evaluation/roc_curve.svg")
    roc_plot
    return


@app.cell
def _(goals, goals_post_shot, post_shot, pre_shot, sportlogiq):
    pre_prob_true, pre_prob_pred = calibration_curve(goals, pre_shot, n_bins=6)
    post_prob_true, post_prob_pred = calibration_curve(goals_post_shot, post_shot, n_bins=6)
    sportlogiq_prob_true, sportlogiq_prob_pred = calibration_curve(goals, sportlogiq, n_bins=6)

    df_pre_cal = pl.DataFrame({
        'model': 'Pre-Shot',
        'mean_predicted_probability': pre_prob_pred,
        'fraction_of_positives': pre_prob_true
    })
    df_post_cal = pl.DataFrame({
        'model': 'Post-Shot',
        'mean_predicted_probability': post_prob_pred,
        'fraction_of_positives': post_prob_true
    })
    df_sportlogiq_cal = pl.DataFrame({
        'model': 'Sportlogiq',
        'mean_predicted_probability': sportlogiq_prob_pred,
        'fraction_of_positives': sportlogiq_prob_true
    })

    calibration_df = pl.concat([df_pre_cal, df_post_cal, df_sportlogiq_cal])

    calibration_plot = (
        ggplot(calibration_df, aes(x="mean_predicted_probability", y="fraction_of_positives", color="model"))
        + p9.geom_abline(slope=1, intercept=0, linetype="dashed", alpha=0.6)
        + geom_line()
        + geom_point()
        + p9.xlim(0,1)
        + p9.ylim(0,1)
        + labs(x="Mean Predicted Probability", y="Fraction of Positives", color="Model")
        + theme_bw(base_size=12)
    )

    # calibration_plot.save("plots/evaluation/calibration.svg")
    calibration_plot
    return


if __name__ == "__main__":
    app.run()
