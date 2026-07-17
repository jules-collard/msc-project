import polars as pl
from polars import col as c
from polars import selectors as cs

from utils import cohens_kappa

def evaluate_shot_detection(shots_with_features: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to evaluate the performance of the shot detection algorithm. Takes a dataframe of shots with features
    (as output by calculate_shot_detection) and calculates the accuracy, precision, recall, and F1 score of the
    algorithm, as well as the proportion of missing data for each shot.
    
    Returns a dataframe with the evaluation metrics.
    """
    
    shots_with_on_target = (
        shots_with_features
        .with_columns(
            on_target = (c('outcome') == 'successful'),
            est_on_target = (c('goalline_y').is_between(-3, 3) & c('goalline_z').is_between(0, 4))
        )
    )

    missing_data_summary = (
        shots_with_on_target
        .select(
            pl.any_horizontal(c('shot_time', 'shot_x', 'shot_y', 'shot_z').is_null()).mean().alias('shot_missing'),
            pl.any_horizontal(cs.starts_with('traj').is_null()).mean().alias('trajectory_missing'),
            pl.any_horizontal(cs.starts_with('goalline').is_null()).mean().alias('projection_missing')
        )
    )

    classification_summary = (
        shots_with_on_target
        # Only evaluate unblocked shots & shots with estimated on-target information
        .filter(
            c('type').str.contains('blocked').not_(),
            c('est_on_target').is_not_null()
        ).select(c('on_target', 'est_on_target'))
        .with_columns(
            true_positive = c('on_target') & c('est_on_target'),
            false_positive = c('on_target').not_() & c('est_on_target'),
            true_negative = c('on_target').not_() & c('est_on_target').not_(),
            false_negative = c('on_target') & c('est_on_target').not_()
        ).select(
            c('true_positive').sum().alias('true_positive'),
            c('false_positive').sum().alias('false_positive'),
            c('true_negative').sum().alias('true_negative'),
            c('false_negative').sum().alias('false_negative')
        ).with_columns(
            accuracy = (c('true_positive') + c('true_negative')) / pl.sum_horizontal(pl.all()),
            precision = c('true_positive') / (c('true_positive') + c('false_positive')),
            recall = c('true_positive') / (c('true_positive') + c('false_negative')),
            cohen_kappa = cohens_kappa('true_positive', 'true_negative', 'false_positive', 'false_negative')
        ).with_columns(
            f1_score = 2 * (c('precision') * c('recall')) / (c('precision') + c('recall'))
        ).drop(cs.starts_with('true', 'false'))
    )

    return pl.concat([missing_data_summary, classification_summary], how='horizontal')