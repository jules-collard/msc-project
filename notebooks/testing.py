import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl


@app.cell
def _():
    (
        pl.read_json(
            "data/20260521/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_1.json"
        )
    )
    return


if __name__ == "__main__":
    app.run()
