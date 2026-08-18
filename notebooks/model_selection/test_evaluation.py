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
    shot_data = batch_read_shot_data("/output/shot_data/20252026/*.parquet")
    pre_shot_xg = pl.scan_parquet("/output/predictions/20252026/pre_shot_1308.parquet")
    post_shot_xg = pl.scan_parquet("/output/predictions/20252026/post_shot_1308.parquet")
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
        ).rename({'pre_shot_xg':'sportlogiq_xg'})
        .with_columns(pl.when(c('type').str.contains('blocked')).then(0).otherwise(c('post_shot')).alias('post_shot'))
        # .select(c('game_id', 'period', 'shot_id', 'goal', 'sportlogiq_xg', 'pre_shot', 'post_shot'))
        .collect()
    )
    return (data,)


@app.function
def get_predictions(data: pl.DataFrame):
    base_rate = data.select(c('goal')).to_series().mean()
    goals = data.select(c('goal')).to_series().to_list()
    sportlogiq = data.select(c('sportlogiq_xg')).to_series().to_list()
    pre_shot = data.select(c('pre_shot')).to_series().to_list()
    post_shot = data.select(c('post_shot')).to_series().to_list()

    return goals, base_rate, sportlogiq, pre_shot, post_shot


@app.cell
def _(data):
    unblocked = data.filter(c('post_shot').is_not_null(), c('type').str.contains('blocked').not_())

    goals, base_rate, sportlogiq, pre_shot, _ = get_predictions(data)
    unblocked_goals, unblocked_base_rate, unblocked_sportlogiq, unblocked_pre_shot, unblocked_post_shot = get_predictions(unblocked)


    model_metrics = pl.DataFrame({
        'shots': ['All Shots', 'All Shots', 'All Shots',
                  'Unblocked Shots', 'Unblocked Shots', 'Unblocked Shots', 'Unblocked Shots'],
        'model': ['Random', 'Sportlogiq', 'Pre-Shot', 'Random', 'Sportlogiq', 'Pre-Shot', 'Post-Shot'],
        'prauc': [
            base_rate,
            average_precision_score(goals, sportlogiq),
            average_precision_score(goals, pre_shot),
            unblocked_base_rate,
            average_precision_score(unblocked_goals, unblocked_sportlogiq),
            average_precision_score(unblocked_goals, unblocked_pre_shot),
            average_precision_score(unblocked_goals, unblocked_post_shot),
        ],
        'rocauc': [
            0.5,
            roc_auc_score(goals, sportlogiq),
            roc_auc_score(goals, pre_shot),
            0.5,
            roc_auc_score(unblocked_goals, unblocked_sportlogiq),
            roc_auc_score(unblocked_goals, unblocked_pre_shot),
            roc_auc_score(unblocked_goals, unblocked_post_shot),
        ],
        'ece': [
            None,
            classwise_ece(sportlogiq, goals, method='frequency'),
            classwise_ece(pre_shot, goals, method='frequency'),
            None,
            None,
            None,
            classwise_ece(unblocked_post_shot, unblocked_goals, method='frequency'),
        ]
    })
    return (
        base_rate,
        goals,
        model_metrics,
        pre_shot,
        sportlogiq,
        unblocked_base_rate,
        unblocked_goals,
        unblocked_post_shot,
    )


@app.cell
def _(model_metrics):
    metrics_table = (
        GT(model_metrics)
        .tab_header(title="Model Evaluation Metrics")
        .tab_source_note("Scores are evaluated on test set (2025-26 Season)")
        .cols_label(shots="Shots", model="Model", prauc="PR-AUC", rocauc="ROC-AUC", ece="ECE")
        .tab_stub(rowname_col='model', groupname_col='shots')
        .sub_missing()
        .fmt_number(['prauc', 'rocauc', 'ece'], decimals=3)
    )

    metrics_table
    # print(metrics_table.as_latex())
    return


@app.cell
def _(goals, pre_shot, sportlogiq, unblocked_goals, unblocked_post_shot):
    pre_precision, pre_recall, _ = precision_recall_curve(goals, pre_shot, drop_intermediate=True)
    post_precision, post_recall, _ = precision_recall_curve(unblocked_goals, unblocked_post_shot, drop_intermediate=True)
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
def _(goals, pre_shot, sportlogiq, unblocked_goals, unblocked_post_shot):
    pre_fpr, pre_tpr, __ = roc_curve(goals, pre_shot, drop_intermediate=True)
    post_fpr, post_tpr, __ = roc_curve(unblocked_goals, unblocked_post_shot, drop_intermediate=True)
    sportlogiq_fpr, sportlogiq_tpr, __ = roc_curve(goals, sportlogiq, drop_intermediate=True)
    return post_fpr, post_tpr, pre_fpr, pre_tpr, sportlogiq_fpr, sportlogiq_tpr


@app.cell
def _(
    base_rate,
    post_precision,
    post_recall,
    pre_precision,
    pre_recall,
    sportlogiq_precision,
    sportlogiq_recall,
    unblocked_base_rate,
):
    df_pre = pl.DataFrame({
        'model': 'Pre-Shot',
        'data': 'All Shots',
        'precision': pre_precision,
        'recall': pre_recall,
        'base_rate': base_rate
    })

    df_sportlogiq = pl.DataFrame({
        'model': 'Sportlogiq',
        'data': 'All Shots',
        'precision': sportlogiq_precision,
        'recall': sportlogiq_recall,
        'base_rate': base_rate
    })

    df_post = pl.DataFrame({
        'model': 'Post-Shot',
        'data': 'Unblocked Shots',
        'precision': post_precision,
        'recall': post_recall,
        'base_rate': unblocked_base_rate
    })

    pr_df = pl.concat([df_pre, df_sportlogiq, df_post])

    pr_plot = (
        ggplot(pr_df, aes(x='recall', y='precision', color='model'))
        + geom_line()
        + p9.geom_hline(aes(yintercept='base_rate', linetype='data'))
        + theme_bw(base_size=10)
        + p9.xlim(0,1)
        + p9.ylim(0,1)
        + labs(y="Precison", x="Recall", color="Model", linetype="Data")
    )

    # pr_plot.save("plots/evaluation/pr_curve.svg")
    pr_plot
    return


@app.cell
def _(post_fpr, post_tpr, pre_fpr, pre_tpr, sportlogiq_fpr, sportlogiq_tpr):
    df_roc_pre = pl.DataFrame({
        'model': 'Pre-Shot',
        'data': 'All Shots',
        'fpr': pre_fpr,
        'tpr': pre_tpr,
    })

    df_roc_sportlogiq = pl.DataFrame({
        'model': 'Sportlogiq',
        'data': 'All Shots',
        'fpr': sportlogiq_fpr,
        'tpr': sportlogiq_tpr,
    })

    df_roc_post = pl.DataFrame({
        'model': 'Post-Shot',
        'data': 'Unblocked Shots',
        'fpr': post_fpr,
        'tpr': post_tpr,
    })

    roc_df = pl.concat([df_roc_pre, df_roc_sportlogiq, df_roc_post])

    roc_plot = (
        ggplot(roc_df, aes(x='fpr', y='tpr', color='model'))
        + geom_line()
        + p9.geom_abline(linetype='dashed')
        + theme_bw(base_size=10)
        + p9.xlim(0,1)
        + p9.ylim(0,1)
        + labs(y="True Positive Rate", x="False Positive Rate", color="Model", linetype="Data")
    )

    # roc_plot.save("plots/evaluation/roc_curve.svg")
    roc_plot
    return


@app.cell
def _(goals, pre_shot, sportlogiq, unblocked_goals, unblocked_post_shot):
    pre_prob_true, pre_prob_pred = calibration_curve(goals, pre_shot, n_bins=6)
    post_prob_true, post_prob_pred = calibration_curve(unblocked_goals, unblocked_post_shot, n_bins=6)
    sportlogiq_prob_true, sportlogiq_prob_pred = calibration_curve(goals, sportlogiq)

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
        + theme_bw(base_size=10)
    )

    # calibration_plot.save("plots/evaluation/calibration.svg")
    calibration_plot
    return


if __name__ == "__main__":
    app.run()
