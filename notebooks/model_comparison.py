import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import optuna

    from data_readers import batch_read_shot_data
    from models.experiments import ExperimentRunner
    from models.data import prepare_data


@app.cell
def _():
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data)
    return (data,)


@app.cell
def _():
    pre_shot_feature_cols = ["shot_x", "shot_y", "total_pressure", "num_defenders_in_shooting_lane", "num_defenders_in_shadow_lane", "num_pressures_left", "num_pressures_right", "num_pressures_front", "num_pressures_back", "goalie_angle_to_shooter", "goalie_in_shooting_lane", "goalie_in_shadow_lane", "goalie_dist_to_goal", "goalie_speed", "goalie_lateral_speed", "shooter_speed", "shooter_lateral_speed", "shooter_goal_speed", "shooter_dist_to_goal", "shooter_angle_to_goal", "visible_angle"]
    return (pre_shot_feature_cols,)


@app.cell
def _(data, pre_shot_feature_cols):
    experiment = ExperimentRunner(data.collect(), pre_shot_feature_cols, "goal")
    experiment.data_splitter.save_split("models/train_test_20242025.npz")

    train_ids = experiment.data_splitter.train_ids
    test_ids = experiment.data_splitter.test_ids
    return experiment, train_ids


@app.cell
def _(experiment):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = experiment.run_all()
    return


@app.cell
def _():
    info = [
      {
        "framework": "lightgbm",
        "strategy": "RO",
        "best_pr_auc": 0.28942466570729486,
        "best_roc_auc": 0.8238359028731829,
        "best_params": {
          "n_estimators": 817,
          "learning_rate": 0.0060569034878811216,
          "num_leaves": 74,
          "max_depth": 9,
          "min_child_samples": 116,
          "min_child_weight": 0.2376886973031763,
          "subsample": 0.7432088895725554,
          "subsample_freq": 1,
          "colsample_bytree": 0.7568004146151055,
          "sampling_strategy": 0.7768267382433762
        }
      },
      {
        "framework": "lightgbm",
        "strategy": "RU",
        "best_pr_auc": 0.28540487918784685,
        "best_roc_auc": 0.8232562693226019,
        "best_params": {
          "n_estimators": 994,
          "learning_rate": 0.006633779811563442,
          "num_leaves": 91,
          "max_depth": 7,
          "min_child_samples": 42,
          "min_child_weight": 0.0053508924552361364,
          "subsample": 0.7634485981432391,
          "subsample_freq": 4,
          "colsample_bytree": 0.8353318493005764,
          "sampling_strategy": 0.2847389669249985
        }
      },
      {
        "framework": "lightgbm",
        "strategy": "WCE",
        "best_pr_auc": 0.29122875788941344,
        "best_roc_auc": 0.82466519581765,
        "best_params": {
          "n_estimators": 933,
          "learning_rate": 0.003881739874285945,
          "num_leaves": 106,
          "max_depth": 10,
          "min_child_samples": 37,
          "min_child_weight": 0.002642225460810176,
          "subsample": 0.7264171587530601,
          "subsample_freq": 5,
          "colsample_bytree": 0.7004583835039335,
          "wce_weight": 2.2128243586755696
        }
      },
      {
        "framework": "lightgbm",
        "strategy": "None",
        "best_pr_auc": 0.29123829554450814,
        "best_roc_auc": 0.8256420436222006,
        "best_params": {
          "n_estimators": 874,
          "learning_rate": 0.004191549624926894,
          "num_leaves": 32,
          "max_depth": 9,
          "min_child_samples": 117,
          "min_child_weight": 2.1699818841449288,
          "subsample": 0.659720333378772,
          "subsample_freq": 6,
          "colsample_bytree": 0.7693051673033089
        }
      },
      {
        "framework": "xgboost",
        "strategy": "RO",
        "best_pr_auc": 0.28926242241432765,
        "best_roc_auc": 0.8233197046273405,
        "best_params": {
          "n_estimators": 427,
          "learning_rate": 0.007647559378795419,
          "max_depth": 8,
          "min_child_weight": 13,
          "gamma": 0.010391142139897488,
          "subsample": 0.9976911275516169,
          "colsample_bytree": 0.767559063457047,
          "sampling_strategy": 0.22298619743758819
        }
      },
      {
        "framework": "xgboost",
        "strategy": "RU",
        "best_pr_auc": 0.28314505476764223,
        "best_roc_auc": 0.824318461037234,
        "best_params": {
          "n_estimators": 645,
          "learning_rate": 0.004388362512150319,
          "max_depth": 8,
          "min_child_weight": 13,
          "gamma": 0.003754404696025948,
          "subsample": 0.7708443496791256,
          "colsample_bytree": 0.7011522150594187,
          "sampling_strategy": 0.3260474648662209
        }
      },
      {
        "framework": "xgboost",
        "strategy": "WCE",
        "best_pr_auc": 0.2913343124414904,
        "best_roc_auc": 0.8256301619349098,
        "best_params": {
          "n_estimators": 366,
          "learning_rate": 0.009901625227477192,
          "max_depth": 7,
          "min_child_weight": 8,
          "gamma": 0.0004619916199390129,
          "subsample": 0.6557088039388476,
          "colsample_bytree": 0.6568082163382861,
          "wce_weight": 1.6282712202540124
        }
      },
      {
        "framework": "xgboost",
        "strategy": "None",
        "best_pr_auc": 0.2916647252141605,
        "best_roc_auc": 0.82582369508151,
        "best_params": {
          "n_estimators": 627,
          "learning_rate": 0.00782796465807462,
          "max_depth": 7,
          "min_child_weight": 15,
          "gamma": 0.010742722726350757,
          "subsample": 0.9152508353773712,
          "colsample_bytree": 0.838984492046485
        }
      }
    ]
    return (info,)


@app.cell
def _(info):
    pl.from_dicts(info)
    return


@app.cell
def _():
    from sklearn.metrics import average_precision_score, roc_auc_score

    return (average_precision_score,)


@app.cell
def _(average_precision_score, data, experiment, train_ids):
    average_precision_score(experiment.y_train, data.filter(pl.col('game_id').is_in(train_ids)).select(pl.col('pre_shot_xg')).collect().to_series())
    return


if __name__ == "__main__":
    app.run()
