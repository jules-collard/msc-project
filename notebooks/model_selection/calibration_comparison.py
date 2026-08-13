import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import json

    import numpy as np
    import polars as pl
    from polars import col as c
    from sklearn.model_selection import train_test_split
    from sklearn.frozen import FrozenEstimator
    from sklearn.calibration import calibration_curve
    import lightgbm as lgb
    import plotnine as p9
    from plotnine import ggplot, aes, geom_line, geom_point, labs
    from betacal import BetaCalibration
    from calibra.errors import classwise_ece

    from data_readers import batch_read_shot_data
    from models.data import DataSplitter, prepare_data, polars_to_pandas
    from models.features import pre_shot_features
    from models.training import ModelTrainer
    from models.calibration import LossCalibratedClassifier, Calibrator


@app.cell
def _():
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data).collect()

    split = np.load("models/train_test_20242025.npz", allow_pickle=True)
    train_ids = split['train_ids']
    test_val_ids = split['test_ids']
    return data, test_val_ids, train_ids


@app.cell
def _(data, test_val_ids, train_ids):
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=56)

    X_train, y_train = DataSplitter.extract_features_and_target(data.filter(c('game_id').is_in(train_ids)), pre_shot_features, "goal")
    X_val, y_val = DataSplitter.extract_features_and_target(data.filter(c('game_id').is_in(val_ids)), pre_shot_features, "goal")
    X_test, y_test = DataSplitter.extract_features_and_target(data.filter(c('game_id').is_in(test_ids)), pre_shot_features, "goal")

    X_train = polars_to_pandas(X_train)
    X_val = polars_to_pandas(X_val)
    X_test = polars_to_pandas(X_test)
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell
def _():
    with open("models/pre_shot/pre_shot_lightgbm_params.json", "r") as f:
        params = json.load(f)

    wce_weight = params.pop('wce_weight')
    params['scale_pos_weight'] = wce_weight
    return (params,)


@app.cell
def _(X_test, X_train, X_val, params, y_train, y_val):
    base_clf = lgb.LGBMClassifier(objective='binary', random_state=78, n_estimators=1500, verbosity=-1, **params)
    base_clf.fit(X_train, y_train)

    loss_calibrated_clf = LossCalibratedClassifier(FrozenEstimator(base_clf))
    loss_calibrated_clf.fit(X_train, y_train)

    isotonic_clf = Calibrator.calibrate(loss_calibrated_clf, X_val, y_val, method='isotonic')
    sigmoid_clf = Calibrator.calibrate(loss_calibrated_clf, X_val, y_val, method='sigmoid')

    bc = BetaCalibration()
    probs = loss_calibrated_clf.predict_proba(X_val)[:,1]
    bc.fit(probs.reshape(-1,1), y_val)

    bc_pred = bc.predict(loss_calibrated_clf.predict_proba(X_test)[:,1].reshape(-1,1))
    return base_clf, bc_pred, isotonic_clf, loss_calibrated_clf, sigmoid_clf


@app.function
def get_calibration_df(classifier, X_test, y_test, name: str, n_bins=6):
    pred = classifier.predict_proba(X_test)[:,1]
    prob_true, prob_pred = calibration_curve(y_test, pred, n_bins=n_bins)
    return pl.DataFrame({
        'mean_predicted_probability': prob_pred,
        'fraction_of_positives': prob_true,
        'model': name
    })


@app.function
def get_classwise_ece(classifers: dict, X_test, y_test):
    results = {}
    for name, classifier in classifers.items():
        preds = classifier.predict_proba(X_test)[:,1]
        ece = classwise_ece(preds, y_test, num_bins=20, method='width')
        results[name] = ece

    return results


@app.cell
def _(bc_pred, y_test):
    bc_prob_true, bc_prob_pred = calibration_curve(y_test, bc_pred, n_bins=6)
    bc_df = pl.DataFrame({
        'mean_predicted_probability': bc_prob_pred,
        'fraction_of_positives': bc_prob_true,
        'model': 'Beta'
    })
    return (bc_df,)


@app.cell
def _(
    X_test,
    base_clf,
    bc_df,
    isotonic_clf,
    loss_calibrated_clf,
    sigmoid_clf,
    y_test,
):
    results = pl.concat([
        get_calibration_df(base_clf, X_test, y_test, 'Uncalibrated'),
        get_calibration_df(loss_calibrated_clf, X_test, y_test, 'Loss-Calibrated'),
        get_calibration_df(isotonic_clf, X_test, y_test, 'Isotonic'),
        get_calibration_df(sigmoid_clf, X_test, y_test, 'Sigmoid'),
        bc_df
    ]).with_columns(c('model').cast(pl.Categorical()))
    return (results,)


@app.cell
def _(results):
    (
        ggplot(results, aes(x="mean_predicted_probability", y="fraction_of_positives", fill="model", color="model"))
        + p9.geom_abline(intercept=0, slope=1, alpha=0.5, linetype="dashed")
        + geom_line(alpha=0.8)
        + geom_point()
        + p9.theme_bw(base_size=10)
        + labs(x='Predicted Probability', y='Fraction Of Positives', fill="Calibration Model", color="Calibration Model")
    )
    return


@app.cell
def _(
    X_test,
    base_clf,
    bc_pred,
    isotonic_clf,
    loss_calibrated_clf,
    sigmoid_clf,
    y_test,
):
    ece_results = get_classwise_ece({
        'Uncalibrated': base_clf,
        'Loss-Calibrated': loss_calibrated_clf,
        'Isotonic': isotonic_clf,
        'Sigmoid': sigmoid_clf,
    }, X_test, y_test)

    ece_results['Beta'] = classwise_ece(bc_pred, y_test, num_bins=6)

    pl.DataFrame(ece_results)
    return


if __name__ == "__main__":
    app.run()
