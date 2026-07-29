from dataclasses import dataclass
from functools import cached_property

import polars as pl
from polars import col as c
import polars.selectors as cs

from processing.tracking import calculate_elapsed_time, adjust_vectors
from pre_shot.geometry import angle_to_shooter
from pre_shot.pressure import pressure, pressure_direction
from pre_shot.lanes import inside_shooting_lane, inside_shadow_lane
from utils import distance_2d, magnitude_2d, distance_to_point_2d


@dataclass
class PreShotData:

    shots: pl.LazyFrame
    player_tracking: pl.LazyFrame
    player_id_mapping: pl.DataFrame

    shadow_expansion: float = 3.0
    d_front: float = 18.0
    d_back: float = 5.0
    q: float = 1.0
    n: float = 5.0
    c: float = 5.0

    @cached_property
    def shots_prepared(self) -> pl.LazyFrame:
        return (
            self.shots
            .with_columns( # Fill missing detection with event details
                c('shot_time').fill_null(c('elapsed_time')),
                c('shot_x').fill_null(c('x_adj_coord')),
                c('shot_y').fill_null(c('y_adj_coord'))
            )
        )

    @cached_property
    def player_tracking_prepared(self) -> pl.LazyFrame:
        return (
            self.player_tracking
            .join(
                self.player_id_mapping.select(c('EntityOfficialID', 'SportlogiqPlayerID')).lazy(),
                left_on='entity_official_id',
                right_on='EntityOfficialID',
                how='left'
            ).with_columns(elapsed_time = calculate_elapsed_time())
            .drop(c('segment_idx', 'clock_state'), cs.starts_with('kappa'))
        )

    def defender_data(self) -> pl.LazyFrame:
        return (
            self.shots_prepared
            .with_columns(
                defender_id = pl.concat_list(c('opposing_team_forwards_on_ice_refs', 'opposing_team_defencemen_on_ice_refs'))
            ).explode('defender_id', empty_as_null=True)
            .sort(c('game_id', 'period', 'defender_id', 'shot_time'))
            .join_asof(
                self.player_tracking_prepared.sort(c('game_id', 'period', 'SportlogiqPlayerID', 'elapsed_time')),
                left_on='shot_time',
                right_on='elapsed_time',
                by_left=['game_id', 'period', 'defender_id'],
                by_right=['game_id', 'period', 'SportlogiqPlayerID'],
                tolerance=0.15,
                check_sortedness=False
            ).with_columns(
                adjust_vectors(c('x', 'y', 'vx', 'vy', 'ax', 'ay'))
            ).with_columns(
                angle_to_shooter(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj')),
                inside_shooting_lane(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj')),
                inside_shadow_lane(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj'), lane_expansion=self.shadow_expansion),
                dist_to_shooter = distance_2d(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj')),
            ).with_columns(
                pressure(c('angle_to_shooter'), c('dist_to_shooter'), d_front=self.d_front, d_back=self.d_back, q=self.q, n=self.n, c=self.c)
            ).with_columns(
                pl.when(c('pressure') > 0).then(pressure_direction(c('angle_to_shooter'))).otherwise(None).alias('pressure_direction')
            )
            # .select(
            #     c('game_id', 'period', 'shot_id', 'defender_id',
            #       'dist_to_shooter', 'angle_to_shooter', 'pressure', 'pressure_direction', 'inside_shooting_lane', 'inside_shadow_lane')
            # )
        )

    def goalie_data(self) -> pl.LazyFrame:
        return (
            self.shots_prepared
            .sort(c('game_id', 'period', 'opposing_team_goalie_on_ice_ref', 'shot_time'))
            .join_asof(
                self.player_tracking_prepared.sort(c('game_id', 'period', 'SportlogiqPlayerID', 'elapsed_time')),
                left_on='shot_time',
                right_on='elapsed_time',
                by_left=['game_id', 'period', 'opposing_team_goalie_on_ice_ref'],
                by_right=['game_id', 'period', 'SportlogiqPlayerID'],
                tolerance=0.15,
                check_sortedness=False
            ).with_columns(
                adjust_vectors(c('x', 'y', 'vx', 'vy', 'ax', 'ay')),
                goalie_speed = magnitude_2d(c('vx'), c('vy')),
                lateral_speed = c('vy').abs()
            ).with_columns(
                angle_to_shooter(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj')).alias('goalie_angle_to_shooter'),
                inside_shooting_lane(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj')).alias('goalie_in_shooting_lane'),
                inside_shadow_lane(c('shot_x'), c('shot_y'), c('x_adj'), c('y_adj'), lane_expansion=self.shadow_expansion).alias('goalie_in_shadow_lane'),
                distance_to_point_2d(c('x_adj'), c('y_adj'), 89, 0).alias('goalie_dist_to_goal')
            )
        )

    def full_output(self) -> pl.LazyFrame:
        defender_summary = (
            self.defender_data()
            .group_by('game_id', 'period', 'shot_id')
            .agg(
                c('pressure').sum().alias('total_pressure'),
                c('inside_shooting_lane').sum().alias('num_defenders_in_shooting_lane'),
                c('inside_shadow_lane').sum().alias('num_defenders_in_shadow_lane'),
                (c('pressure_direction') == 'left').sum().alias('num_pressures_left'),
                (c('pressure_direction') == 'right').sum().alias('num_pressures_right'),
                (c('pressure_direction') == 'front').sum().alias('num_pressures_front'),
                (c('pressure_direction') == 'back').sum().alias('num_pressures_back'),
                pl.len().alias('num_tracked_defenders')
            )
        )

        return (
            self.shots_prepared
            .join(
                defender_summary,
                on=['game_id', 'period', 'shot_id'],
                how='left'
            ).join(
                self.goalie_data().select(
                    c('game_id', 'period', 'shot_id',
                      'goalie_angle_to_shooter', 'goalie_in_shooting_lane', 'goalie_in_shadow_lane', 'goalie_dist_to_goal', 'goalie_speed', 'lateral_speed'),
                    c('x_adj', 'y_adj').name.prefix('goalie_')
                ),
                on=['game_id', 'period', 'shot_id'],
                how='left'
            )
        )