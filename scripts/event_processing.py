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

def add_shot_info(events: pl.DataFrame | pl.LazyFrame):
    """
    Function to extract relevant indicator columns from shot events.

    Creates columns for goal, shot type, net location of shot on target, etc.
    """
    
    return (
        events
        .with_columns(
            shot_goal = pl.when(c('name')=='shot').then(c('flags').list.contains('withgoal')),
            shot_type = pl.when(c('name')=='shot').then(c('flags').list.first()),
            shot_net_location = pl.when(c('name')=='shot', c('outcome') == 'successful').then(c('flags').list.last()),
            shot_pressured = pl.when(c('name')=='shot').then(c('flags').list.contains('withpressure')),
            shot_quickreleased = pl.when(c('name')=='shot').then(c('flags').list.contains('quickrelease')),
            shot_onetimer = pl.when(c('name')=='shot').then(c('flags').list.contains('1timer')),
            shot_rebound = pl.when(c('name')=='shot').then(c('flags').list.contains('withrebound')),
            shot_on_net = pl.when(c('name')=='shot').then(c('outcome').eq('successful')),
            shot_blocked = pl.when(c('name')=='shot').then(c('type').str.contains('blocked')),
            shot_from_seam_pass = pl.when(c('name')=='shot').then(c('flags').list.contains('seam'))
        )
    )

def add_pass_target(events: pl.DataFrame | pl.LazyFrame):
    """
    Function to extract pass target location, whether successful or not.

    Looks at all events following a pass (until the next pass), and extracts the x_coord and y_coord 
    of either the corresponding reception or failedpasslocation event into pass_target_x_coord and pass_target_y_coord.
    """
    target_events = ['reception', 'failedpasslocation']
    return (
        events
        .with_columns(
            pass_group_id=(c('name') == 'pass').cum_sum().over('current_possession')
        ).with_columns(
            target_x_coord=pl.when(c('name').is_in(target_events)).then(c('x_coord')),
            target_y_coord=pl.when(c('name').is_in(target_events)).then(c('y_coord'))
        ).with_columns(
            pass_target_x_coord=pl.when(c('name')=='pass').then(
                c('target_x_coord').backward_fill().over(['current_possession', 'pass_group_id'])
            ),
            pass_target_y_coord=pl.when(c('name')=='pass').then(
                c('target_y_coord').backward_fill().over(['current_possession', 'pass_group_id'])
            )
        ).drop(c('target_x_coord', 'target_y_coord', 'pass_group_id'))
    )