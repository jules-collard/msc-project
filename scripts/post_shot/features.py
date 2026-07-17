from typing import List
from dataclasses import dataclass
from functools import cached_property

import polars as pl
from polars import col as c

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
            .with_row_index(name='shot_id')
        )

    @cached_property
    def puck_tracking_prepared(self) -> pl.LazyFrame:
        return (
            self.puck_tracking
            .with_columns(elapsed_time = calculate_elapsed_time())
            .filter(
                c('entity_id') == '1'
            )
            .drop(c('entity_id'), c('clock_state'))
        )

    def prepare_shots(self) -> pl.LazyFrame:
        """
        Prepares the shots dataframe by filtering for shot events and adding necessary columns.
        Returns a LazyFrame with shot events and their corresponding shot_id.
        """
        return self.shots
    
    def prepare_puck_tracking(self) -> pl.LazyFrame:
        """
        Prepares the puck tracking dataframe by filtering for puck tracking data and adding necessary columns.
        Returns a LazyFrame with puck tracking data.
        """
        return self.puck_tracking_prepared
    
    def detect_shots(self) -> pl.LazyFrame:
        """
        Detects shots by joining the shots dataframe with the puck tracking dataframe and calculating detection results.
        """
        return calculate_shot_detection(
            self.shots,
            self.puck_tracking_prepared,
            window_size=self.window_size,
            distance_threshold=self.distance_threshold,
            impact_acceleration_threshold=self.impact_acceleration_threshold,
            deflection_angle_threshold=self.deflection_angle_threshold
        )
    
    def with_features(self) -> pl.LazyFrame:
        """
        Returns a LazyFrame with detected shots and their corresponding post-shot features.
        """
        return self.detect_shots().with_columns(shot_features())
    
    def evaluate(self) -> pl.LazyFrame:
        return evaluate_shot_detection(self.full_output())
    
    def full_output(self) -> pl.LazyFrame:
        """
        Returns a LazyFrame with all relevant shot information, including shot details, detection results, and post-shot features.
        """

        detected = self.with_features()

        return (
            self.shots
            .select(
                c('game_id'), c('period'), c('game_time'), c('elapsed_time'), c('shot_id'),
                c('team'), c('player_first_name'), c('player_last_name'), c('player_reference_id'),
                c('x_adj_coord'), c('y_adj_coord'), c('expected_goals_all_shots').alias('pre_shot_xg'),
                c('type'), c('outcome'), c('flags')
            ).join(
                detected,
                on='shot_id',
                how='left'
            )
        )
    
    def model_input(self) -> pl.LazyFrame:
        """
        Returns a LazyFrame with the necessary features for model input, including shot speed, on-goal status, and distances to various goal locations.
        Only includes shots that were detected and are from the offensive zone (x >= 25).
        """
        return (
            self.with_features()
            .drop_nulls(c('shot_speed')) # Only keep shots with valid speed (i.e. shots that were detected)
            .filter(
                c('shot_x') >= 25 # Only o-zone shots
            )
            .select(c('shot_id', 'shot_speed', 'on_goal', 'dist_to_post', 'dist_to_crossbar', 'dist_to_top_corner', 'dist_to_center'))
        )


def shot_features() -> List[pl.Expr]:
    """
    Returns a list of expressions with post-shot features.
    Should be applied to output of post_shot.detection.calculate_shot_detection()
    """

    on_goal = (c('goalline_y').abs() <= 3) & (c('goalline_z') <= 4)
    dist_to_post = 3 - c('goalline_y').abs()
    dist_to_crossbar = 4 - c('goalline_z')
    dist_to_top_corner = distance_to_point_2d(c('goalline_y').abs(), c('goalline_z'), 3, 4)
    dist_to_top_corner = ( # Distance to top corner is negative when not on target
        pl.when(on_goal)
        .then(dist_to_top_corner)
        .otherwise(-dist_to_top_corner)
    )
    dist_to_center = distance_to_point_2d(c('goalline_y'), c('goalline_z'), 0, 2)
    nearest_post_y = pl.when(c('goalline_y') < 0).then(-3).otherwise(3)

    return [
        on_goal.alias('on_goal'),
        dist_to_post.alias('dist_to_post'),
        dist_to_crossbar.alias('dist_to_crossbar'),
        dist_to_top_corner.alias('dist_to_top_corner'),
        dist_to_center.alias('dist_to_center'),
        nearest_post_y.alias('nearest_post_y')
    ]
