import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import polars as pl
    from polars import col as c
    import polars.selectors as cs

    from parquet_helpers import EntityTrackingReader

    return EntityTrackingReader, c, cs, json, pl


@app.cell
def _(EntityTrackingReader):
    with open("data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet", mode="rb") as f:
        reader = EntityTrackingReader(f.read())
    return (reader,)


@app.cell
def _(reader):
    tracking = reader.get_table()
    return


@app.cell
def _(json, pl):
    with open("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json", "r") as file:
        events_dict = json.load(file)

    event_data = pl.from_dicts(events_dict.get("events"), infer_schema_length=None)
    return (event_data,)


@app.cell
def _(c, cs, event_data, pl):
    def explode_events(events: pl.DataFrame):
        exploded = (
            events
            .filter(c('name').str.contains('shot'))
            .with_columns(
                with_team = pl.concat_list(c('team_forwards_on_ice_refs', 'team_defencemen_on_ice_refs', 'team_goalie_on_ice_ref')).list.drop_nulls(),
                opposing_team = pl.concat_list(c('opposing_team_forwards_on_ice_refs', 'opposing_team_defencemen_on_ice_refs', 'opposing_team_goalie_on_ice_ref')).list.drop_nulls()
            ).unpivot(
                on=['with_team', 'opposing_team'], 
                variable_name='team_source', 
                value_name='onice_player_id', 
                index=cs.all() - cs.by_name('with_team', 'opposing_team'),
            )
            .explode('onice_player_id')
        )
        return exploded

    exploded_shots = (
        explode_events(event_data.filter(c('name').str.contains('shot')))
        .drop(cs.ends_with('ref'), cs.ends_with('refs'), cs.starts_with('expected_goals'))
    )
    return


if __name__ == "__main__":
    app.run()
