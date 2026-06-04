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

    tracking = reader.get_table()
    return (tracking,)


@app.cell
def _(json, pl):
    with open("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json", "r") as file:
        events_dict = json.load(file)

    events = pl.from_dicts(events_dict.get("events"), infer_schema_length=None).lazy()
    return (events,)


@app.cell
def _(c, cs, pl):
    def explode_events(events: pl.DataFrame | pl.LazyFrame):
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
            .drop(cs.ends_with('on_ice_refs'), cs.by_name('team_goalie_on_ice_ref'))
        )
        return exploded

    return (explode_events,)


@app.cell
def _(c, cs, pl):
    def clean_shots(shots: pl.DataFrame | pl.LazyFrame):
        cleaned_shots = (
            shots
            .with_columns(
                goal = c('flags').list.contains('withgoal'),
                shot_type = c('flags').list.first(),
                net_location = pl.when(c('outcome') == 'successful').then(c('flags').list.last()),
                pressured = c('flags').list.contains('withpressure'),
                quickreleased = c('flags').list.contains('quickrelease'),
                onetimer = c('flags').list.contains('1timer'),
                rebound = c('flags').list.contains('withrebound'),
                on_net = c('outcome').eq('successful'),
                blocked = c('type').str.contains('blocked'),
                seam_pass = c('flags').list.contains('seam')
            ).drop(cs.contains('possession'), cs.starts_with('previous'), cs.starts_with('expected_goals'), cs.by_name('player_jersey'))
        )
        return cleaned_shots

    return (clean_shots,)


@app.cell
def _(c, clean_shots, events, explode_events):
    cleaned_shots = (
        events
        .filter(c('name') == 'shot', c('team_skaters_on_ice') == 5, c('opposing_team_skaters_on_ice') == 5)
        .pipe(clean_shots)
        .pipe(explode_events)
        .collect()
    )
    return


@app.cell
def _(events):
    events.collect()
    return


@app.cell
def _(pl, tracking):
    tracking_pl = pl.from_arrow(tracking)
    return (tracking_pl,)


@app.cell
def _(c, pl, tracking_pl):
    tracking_pl.with_columns(pl.from_epoch(c("ts"), time_unit="s").alias("ts"))
    return


if __name__ == "__main__":
    app.run()
