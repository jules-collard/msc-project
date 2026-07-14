import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, labs, geom_histogram, geom_point, geom_line, geom_text, geom_vline, geom_hline, theme, element_text, scale_x_continuous, scale_y_continuous, facet_wrap, theme_bw

    from data_readers import read_events, read_entity_tracking, read_puck_tracking
    from tracking_processing import derive_game_clock
    from event_processing import add_flip
    from features.puck import calculate_shot_features
    from utils import cohens_kappa


@app.cell
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json").pipe(add_flip)
    shots = events.filter(c('name') == 'shot').sort(c('game_time')).with_row_index(name='shot_id')

    periods = [1,2,3]
    player_tracking = read_entity_tracking(
        "data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet"
    )

    puck_tracking = read_puck_tracking(
        [f"data/one_game/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_{i}.json" for i in periods], periods
    )

    puck_tracking = (
        pl.concat([player_tracking, puck_tracking], how='diagonal_relaxed')
        .pipe(derive_game_clock)
        .filter(c('clock_state') == 1, c('entity_id') == '1')
        .sort(c('game_time'))
        .drop(c('entity_id', 'entity_official_id', 'segment_idx', 'clock_state', 'raw_x', 'raw_y', 'raw_z'))
    )
    return puck_tracking, shots


@app.cell
def _(puck_tracking, shots):
    shot_features = calculate_shot_features(shots, puck_tracking)
    shots_with_features = shots.join(shot_features, on=c('shot_id'), how='left').collect()
    return shot_features, shots_with_features


@app.cell
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(timing_error = c('game_time') - c('shot_time'))
        >> ggplot(aes(x='timing_error'))
        + geom_vline(xintercept=0, linetype='dashed')
        + geom_histogram(bins=25, color='black')
        + p9.theme_bw(base_size=10)
        + labs(x="Estimated Timing Error (seconds)", y="Count", title="Distribution of Estimated Timing Errors for Shots",
              caption="Timing Error = Event Time - Estimated Shot Time")
    )
    return


@app.cell
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(
            x_error = c('x_adj_coord') - c('shot_x'),
            y_error = c('y_adj_coord') - c('shot_y')
        )
        >> ggplot(aes(x='x_error', y='y_error'))
        + geom_vline(xintercept=0, alpha=0.8)
        + geom_hline(yintercept=0, alpha=0.8)
        + p9.geom_density_2d(aes(color='..level..'), levels=9)
        + p9.geom_rug()
        + p9.scale_color_distiller(type='seq', palette='Oranges', direction=1)
        + theme_bw(base_size=10)
        + labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)", title="2D Density of Estimated Shot Location Errors", color="Density", caption="X/Y Coordinate Error = Event Shot Location - Estimated Shot Location")
    )
    return


@app.cell
def _(shot_features):
    shot_features.collect()
    return


@app.cell
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(
            on_target = (c('outcome') == 'successful'),
            est_on_target = (c('goalline_y').is_between(-3, 3) & c('goalline_z').is_between(0, 4))
        )
        .select(
            pl.any_horizontal(c('shot_time', 'shot_x', 'shot_y', 'shot_z').is_null()).mean().alias('shot_missing'),        
            pl.any_horizontal(cs.starts_with('traj').is_null()).mean().alias('trajectory_missing'),
            pl.any_horizontal(cs.starts_with('goalline').is_null()).mean().alias('projection_missing')
        )
    )
    return


@app.cell
def _(shots_with_features):
    (
        shots_with_features
        .with_columns(
            on_target = (c('outcome') == 'successful'),
            est_on_target = (c('goalline_y').is_between(-3, 3) & c('goalline_z').is_between(0, 4))
        )
        # Only evaluate unblocked shots
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
    return


if __name__ == "__main__":
    app.run()
