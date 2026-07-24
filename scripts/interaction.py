import marimo as mo
import polars as pl
from polars import col as c


def game_selectors(game_id_mapping: pl.DataFrame):
    date_selector = mo.ui.date_range.from_series(
        game_id_mapping.select(c('GameDate')).to_series(),
        label="Game Date"
    )

    game_id_selector = mo.ui.multiselect.from_series(
        game_id_mapping.select(c('SportlogiqGameID')).to_series(),
        value=game_id_mapping.select(c('SportlogiqGameID')).to_series(),
        label="Sportlogiq Game ID"
    )

    game_type_selector = mo.ui.multiselect(
        options=["regular", "playoffs"],
        value=["regular", "playoffs"],
        label="Game Type"
    )

    run_button = mo.ui.run_button(kind='success', label="Run Shot Detection")

    return date_selector, game_id_selector, game_type_selector, run_button

def display_game_selectors(date_selector, game_id_selector, game_type_selector, run_button):
    return mo.vstack([
        mo.hstack([date_selector, game_id_selector, game_type_selector]),
        run_button,
    ])