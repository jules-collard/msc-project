import polars as pl
from polars import col as c
from polars import selectors as cs

def explode_events(events: pl.DataFrame | pl.LazyFrame):
    """
    Function to lengthen event data for joining with tracking data.

    Each event is duplicated, with one row per player on ice per event.
    """

    exploded = (
        events
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

def clean_shots(shots: pl.DataFrame | pl.LazyFrame):
    """
    Function to extract relevant indicator columns from shot events.

    Creates columns for goal, shot type, net location of shot on target, etc.
    """

    try:
        assert shots.select(c('name') == 'shot').to_series().all()
    except AttributeError: # If LazyFrame
        assert shots.select(c('name') == 'shot').collect().to_series().all()
    
    return (
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