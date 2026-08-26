import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, geom_point, geom_smooth, labs, theme_bw
    from mizani.formatters import percent_format

    from data_readers import batch_read_shot_data
    from models.data import prepare_data


@app.cell
def _():
    shots_2425 = batch_read_shot_data("/output/shot_data/20242025/*.parquet").with_columns(season=pl.lit("20242025"))
    shots_2526 = batch_read_shot_data("/output/shot_data/20252026/*.parquet").with_columns(season=pl.lit("20252026"))

    shots = pl.concat([shots_2425, shots_2526], how="vertical").collect()
    return (shots,)


@app.cell
def _(shots):
    season_pairs = (
        shots
        .with_columns(
            pl.when(c('position') == 'D').then(pl.lit('Defenders')).otherwise(pl.lit('Forwards')).alias('position_group')
        )
        .group_by("player_reference_id", "position_group", "season")
        .agg(
            pl.len().alias('shots'),
            (c('type').str.contains('blocked')).sum().alias('blocked_shots')
        ).filter(c('shots') >= 50)
        .with_columns(
            blocked_shot_rate = c('blocked_shots') / c('shots')
        ).pivot(
            on="season",
            values="blocked_shot_rate",
            index=["player_reference_id", "position_group"]
        ).drop_nulls()
    )
    return (season_pairs,)


@app.cell
def _(season_pairs):
    r_squared = season_pairs.select(pl.corr("20242025", "20252026").pow(2)).item()
    return (r_squared,)


@app.cell
def _(r_squared, season_pairs):
    corr_plot = (
        ggplot(season_pairs, aes(x="20242025", y="20252026", colour="position_group"))
        + geom_point()
        + geom_smooth(method="lm", se=False, colour="black")
        + p9.geom_label(label=f"R^2={r_squared:.2f}", x=0.2, y=0.45, colour="black")
        + theme_bw(base_size=12)
        + labs(x="2024-2025 Blocked Shot Rate", y="2025-2026 Blocked Shot Rate", colour="Position", caption="min. 50 Shot Attempts")
        + p9.coord_fixed(xlim=(0.08,0.52), ylim=(0.08,0.52))
        + p9.scale_x_continuous(labels=percent_format())
        + p9.scale_y_continuous(labels=percent_format())
    )

    # corr_plot.save("plots/metrics/blocked_shot_rates.svg")
    corr_plot
    return


if __name__ == "__main__":
    app.run()
