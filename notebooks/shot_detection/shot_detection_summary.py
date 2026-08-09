import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_vline, geom_hline, geom_density, theme_bw, labs

    from data_readers import batch_read_shot_data
    from models.data import DataSplitter, prepare_data

    return (
        DataSplitter,
        aes,
        batch_read_shot_data,
        c,
        cs,
        geom_hline,
        geom_vline,
        ggplot,
        labs,
        p9,
        pl,
        prepare_data,
        theme_bw,
    )


@app.cell
def _(DataSplitter, batch_read_shot_data, cs, prepare_data):
    data = batch_read_shot_data("/output/shot_data/20242025/*.parquet").pipe(prepare_data)

    splitter = DataSplitter(data.collect(), cs.all().exclude('goal'), 'goal', split_path="models/train_test_20242025.npz")
    return (splitter,)


@app.cell
def _(splitter):
    data_train, _y_train, _X_test, _y_test, _groups = splitter.get_split_data()
    return (data_train,)


@app.cell
def _(aes, c, data_train, geom_vline, ggplot, labs, p9, theme_bw):
    timing_plot = (
        data_train
        .with_columns(timing_error = c('elapsed_time') - c('shot_time'))
        >> ggplot(aes(x='timing_error'))
        + geom_vline(xintercept=0, linetype='dashed')
        # + geom_density(fill='lightblue', alpha=0.5)
        + p9.geom_histogram(binwidth=0.05, color='black', fill='lightgrey')
        + theme_bw(base_size=10)
        + p9.xlim((-0.8, 0.8))
        + labs(x="Estimated Timing Error (seconds)", y="No. Shots",
              caption="2024-25 Training Set")
    )

    timing_plot.save("plots/shot_detection/timing_error_distribution.svg")
    return


@app.cell
def _(aes, c, data_train, geom_hline, geom_vline, ggplot, labs, p9, theme_bw):
    location_plot = (
        data_train
        .with_columns(
            x_error = c('x_adj_coord') - c('shot_x'),
            y_error = c('y_adj_coord') - c('shot_y')
        )
        >> ggplot(aes(x='x_error', y='y_error'))
        + geom_vline(xintercept=0, alpha=0.8)
        + geom_hline(yintercept=0, alpha=0.8)
        # + p9.geom_point(alpha=0.1)
        + p9.stat_density_2d(aes(fill='stat(level)'), geom='polygon', levels=10, alpha=0.8)
        + p9.geom_density_2d(levels=10, alpha=0.5)
        + p9.scale_fill_distiller(type='seq', palette='Greys', direction=1)
        + theme_bw(base_size=10)
        + labs(x="Estimated X Coordinate Error (ft)", y="Estimated Y Coordinate Error (ft)", fill="Density", caption="2024-25 Training Set")
        + p9.theme(aspect_ratio=1)
    )

    location_plot.save("plots/shot_detection/location_error_distribution.svg")
    return


@app.cell
def _(c, cs, data_train, pl):
    (
        data_train
        .select(
            (1 - pl.any_horizontal(c('shot_time', 'shot_x', 'shot_y', 'shot_z').is_null()).mean()).alias('shot_start_detected'),
            (1 - pl.any_horizontal(cs.starts_with('traj').is_null()).mean()).alias('trajectory_detected'),
            (1 - pl.any_horizontal(cs.starts_with('goalline').is_null()).mean()).alias('projection')
        )
    )
    return


if __name__ == "__main__":
    app.run()
