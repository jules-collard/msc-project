from typing import Dict

import polars as pl
from polars import col as c
from polars import selectors as cs

from processing.tracking import adjust_vectors

def explode_events(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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
            value_name='onice_player_ref', 
            index=cs.all() - cs.by_name('with_team', 'opposing_team'),
        )
        .explode('onice_player_ref')
        .drop(cs.ends_with('on_ice_refs'), cs.by_name('team_goalie_on_ice_ref'))
    )
    return exploded

def add_shot_info(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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

def add_pass_target(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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

def add_carry_info(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to extract endpoint of carry events.

    Not extensive - sometimes will be incorrect due to other events being present.
    """

    return (
        events
        .with_columns(
            carry_target_x_coord=pl.when(c('name') == 'carry').then(c('x_coord').shift(-1)),
            carry_target_y_coord=pl.when(c('name') == 'carry').then(c('y_coord').shift(-1))
        )
    )

def remove_non_viz_events(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to remove events that are not useful for visualisation.
    """
    return (
        events
        .filter(
            c('name').is_in(['pressure', 'assist', 'goal', 'failedpasslocation', 'controlledentryagainst', 'dumpinagainst']).not_(),
            c('name').eq('faceoff').and_(c('zone').is_not_null()).not_(), # remove faceoff win/loss rows
        )
    )

def add_flip(events: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to identify whether raw coordinates require flipping to recover adjusted coordinates.
    """
    return (
        events
        .with_columns(
            flip = (c('x_coord').ne(c('x_adj_coord'))).or_(c('y_coord').ne(c('y_adj_coord')))
        )
    )

def join_tracking(
    events: pl.DataFrame | pl.LazyFrame,
    tracking: pl.DataFrame | pl.LazyFrame,
    mapping: Dict[str, str]
) -> pl.DataFrame | pl.LazyFrame:
    """
    Function to join tracking data to events. Tracking data MUST have game_time field.
    For each event and each on-ice player, finds latest tracking observation before the
    event, within 0.15s. Tracking data is also normalised to match adj_coord alignment.
    
    Returns dataframe/lazyframe with one row per on-ice player per event.
    """

    exploded_events = (
        events
        .sort(c('game_time'))
        .pipe(add_flip)
        .pipe(explode_events)
        .with_columns(
            c('onice_player_ref').replace_strict(mapping), 
            c('opposing_team_goalie_on_ice_ref').replace_strict(mapping, default=None)
        )
    )
    cleaned_tracking = (
        tracking
        .drop('ts', 'period', 'segment_idx', 'clock_state', cs.starts_with('raw'), 'smt_speed', cs.starts_with('kappa'))
        .sort(c('game_time'))
    )
    return (
        exploded_events
        .join_asof(
            cleaned_tracking, 
            on='game_time',
            by_left='onice_player_ref', 
            by_right='entity_official_id',
            tolerance=0.15,
            check_sortedness=False
        ).pipe(adjust_vectors)
    )

def timecode_to_seconds(expr: pl.Expr | str = 'timecode') -> pl.Expr:
    """
    Function to convert timecode string (MM:SS) to elapsed seconds from period start.
    """
    if isinstance(expr, str):
        expr = c(expr)

    parts = (
        expr
        .str.split_exact(':', 3)
        .struct.rename_fields(['timecode_hr', 'timecode_min', 'timecode_sec', 'timecode_frame'])
    )

    return (
        parts.struct.field('timecode_hr').cast(pl.Int32) * 3600 +
        parts.struct.field('timecode_min').cast(pl.Int32) * 60 +
        parts.struct.field('timecode_sec').cast(pl.Int32) +
        parts.struct.field('timecode_frame').cast(pl.Float64) / 30
    )
