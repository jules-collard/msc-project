from typing import List, Tuple
from dataclasses import dataclass
from functools import cached_property

import polars as pl
from polars import col as c
import numpy as np

from processing.events import timecode_to_seconds, extract_flip
from processing.tracking import calculate_elapsed_time
from post_shot.detection import calculate_shot_detection
from post_shot.evaluation import evaluate_shot_detection
from utils import distance_to_point_2d

@dataclass
class PostShotData:
    """
    Class to handle post-shot data processing, including shot detection, feature extraction, and evaluation.
    Takes in events and puck tracking data, and provides methods to prepare the data, detect shots, extract features,
    and evaluate the shot detection algorithm.

    window_size: float, optional
        The size of the window (in seconds) to select tracking data for each shot. Default is 1.6 seconds.
    distance_threshold: float, optional
        The maximum distance (in feet) from the shot location to consider tracking data for the shot.
    impact_acceleration_threshold: float, optional
        The threshold for the goal acceleration to consider the shot as having been blocked/saved/deflected.
    deflection_angle_threshold: float, optional
        The threshold for the change in angle to consider the shot as having been deflected.
    """

    events: pl.LazyFrame
    puck_tracking: pl.LazyFrame
    player_info: pl.LazyFrame
    window_size: float = 1.6
    distance_threshold: float = 8
    impact_acceleration_threshold: float = -800
    deflection_angle_threshold: float = 25

    @cached_property
    def shots(self) -> pl.LazyFrame:
        return (
            self.events
            .with_columns(
                extract_flip(),
                elapsed_time = timecode_to_seconds()
            )
            .filter(c('name') == 'shot')
            .sort(c('game_id'), c('period'), c('elapsed_time'))
            .with_columns(
                c('expected_goals_all_shots').cast(pl.Float32),
                goal = c('flags').list.contains('withgoal')
            ).with_row_index(name='shot_id')
        )

    @cached_property
    def puck_tracking_prepared(self) -> pl.LazyFrame:
        return (
            self.puck_tracking
            .with_columns(elapsed_time = calculate_elapsed_time())
            .drop(c('clock_state'))
        )
    
    def detect_shots(self) -> pl.LazyFrame:
        """
        Returns results of shot detection algorithm, using specified parameters.
        """
        return calculate_shot_detection(
            self.shots,
            self.puck_tracking_prepared,
            window_size=self.window_size,
            distance_threshold=self.distance_threshold,
            impact_acceleration_threshold=self.impact_acceleration_threshold,
            deflection_angle_threshold=self.deflection_angle_threshold
        )
    
    def post_shot_features(self) -> pl.LazyFrame:
        """
        Returns results of shot detection, with derived post-shot features.
        """
        return self.detect_shots().with_columns(shot_features())
    
    def evaluate_detection(self) -> pl.LazyFrame:
        """
        Returns evaluation metrics for the shot detection algorithm.
        """
        return evaluate_shot_detection(self.full_output())
    
    def full_output(self) -> pl.LazyFrame:
        """
        Returns all relevant shot information, including shot details, detection results, and post-shot features.
        """

        detected = self.post_shot_features()
        return (
            self.shots
            .select(
                c('game_id'), c('period'), c('game_time'), c('elapsed_time'), c('shot_id'),
                c('team'), c('player_first_name'), c('player_last_name'), c('player_reference_id'),
                c('opposing_team_goalie_on_ice_ref'), c('team_skaters_on_ice'), c('opposing_team_skaters_on_ice'),
                c('x_adj_coord'), c('y_adj_coord'),
                c('expected_goals_all_shots').alias('pre_shot_xg'),
                c('type'), c('outcome'), c('flags'), c('goal')
            ).join(
                detected,
                on='shot_id',
                how='left'
            ).join( # Add shooter handedness
                self.player_info.select(c('SportlogiqPlayerID'), c('handedness')).rename({'handedness': 'shooter_handedness'}),
                left_on='player_reference_id',
                right_on='SportlogiqPlayerID',
                how='left',
                coalesce=True
            ).join( # Add goalie handedness
                self.player_info.select(c('SportlogiqPlayerID'), c('handedness')).rename({'handedness': 'goalie_handedness'}),
                left_on='opposing_team_goalie_on_ice_ref',
                right_on='SportlogiqPlayerID',
                how='left',
                coalesce=True
            )
        )

    def model_data(self) -> pl.LazyFrame:
        return (
            self.full_output()
            .drop_nulls(c('shot_speed')) # Only keep shots with valid speed (i.e. shots that were detected)
            .filter(
                c('shot_x') >= 25 # Only o-zone shots
            )
        )
    
    def model_input(self) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """
        Returns a LazyFrame with the necessary features for model input. Only includes shots that were
        detected and are from the offensive zone (x >= 25).

        Returns:
        Tuple[pl.DataFrame, np.ndarray, np.ndarray]: A tuple containing:
            - A polars DataFrame with shot IDs
            - A NumPy array with the feature values (X).
            - A NumPy array with the target variable (y), indicating whether the shot resulted in a goal (1) or not (0).
        """
        data = self.model_data()
        ids = data.select(c('shot_id')).collect()
        X = (
            data
            .select(c('pre_shot_xg', 'shot_speed', 'on_goal', 'dist_to_post', 'dist_to_crossbar', 'dist_to_top_corner', 'dist_to_center'))
            .collect()
            .to_numpy()
        )
        y = data.select(c('goal').cast(pl.Int8)).collect().to_numpy().flatten()

        return ids, X, y


def shot_features() -> List[pl.Expr]:
    """
    Returns a list of expressions with post-shot features.
    Should be applied to output of post_shot.detection.calculate_shot_detection()
    """

    on_goal = (c('goalline_y').abs() <= 3) & (c('goalline_z') <= 4)
    dist_to_post = 3 - c('goalline_y').abs()
    dist_to_crossbar = 4 - c('goalline_z')
    dist_to_corner = distance_to_point_2d(c('goalline_y').abs(), c('goalline_z') - 2, 3, 2)
    dist_to_corner = ( # Distance to top corner is negative when not on target
        pl.when(on_goal)
        .then(dist_to_corner)
        .otherwise(-dist_to_corner)
    )
    dist_to_center = distance_to_point_2d(c('goalline_y'), c('goalline_z'), 0, 2)
    nearest_post_y = pl.when(c('goalline_y') < 0).then(-3).otherwise(3)

    polar_dist = distance_to_point_2d(c('goalline_y'), c('goalline_z'), 0, 0)
    polar_angle_raw = pl.arctan2(c('goalline_z'), c('goalline_y')).degrees() - 90
    polar_angle_abs = (pl.arctan2(c('goalline_z'), c('goalline_y')).degrees() - 90).abs()
    polar_angle_near_post = (
        pl.when(c('shot_y') >= 0)
        .then(polar_angle_raw)
        .otherwise(-polar_angle_raw)
    )

    return [
        on_goal.alias('on_goal'),
        dist_to_post.alias('dist_to_post'),
        dist_to_crossbar.alias('dist_to_crossbar'),
        dist_to_corner.alias('dist_to_corner'),
        dist_to_center.alias('dist_to_center'),
        nearest_post_y.alias('nearest_post_y'),
        polar_dist.alias('polar_dist'),
        polar_angle_raw.alias('polar_angle_raw'),
        polar_angle_abs.alias('polar_angle_abs'),
        polar_angle_near_post.alias('polar_angle_near_post')
    ]
