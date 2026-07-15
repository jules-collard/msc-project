import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl


@app.cell
def _():
    (
        pl.scan_parquet(
            "data/*/HOCKEY_NHL_*.parquet",
            include_file_paths="source_path"
        ).with_columns(
            game_id=pl.col("source_path").str.extract(r"/(\d+)/HOCKEY_NHL").cast(pl.String),
            period=pl.col("source_path").str.extract(r"Period_(\d+)").cast(pl.Int32)
        ).collect()
    )
    return


if __name__ == "__main__":
    app.run()
