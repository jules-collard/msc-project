from models.types import ShotType

pre_shot_features = [
    "shot_x",
    "shot_y",
    "total_pressure",
    "num_defenders_in_shooting_lane",
    "num_defenders_in_shadow_lane",
    "num_pressures_left",
    "num_pressures_right",
    "num_pressures_front",
    "num_pressures_back",
    "goalie_angle_to_shooter",
    "goalie_in_shooting_lane",
    "goalie_in_shadow_lane",
    "goalie_dist_to_goal",
    "goalie_speed",
    "goalie_lateral_speed",
    "shooter_speed",
    "shooter_lateral_speed",
    "shooter_goal_speed",
    "shooter_dist_to_goal",
    "shooter_angle_to_goal",
    "visible_angle",
    "shot_type"
]

pre_shot_features_minimal = [
    "total_pressure",
    "num_defenders_in_shadow_lane",
    "goalie_angle_to_shooter",
    "goalie_in_shooting_lane",
    "goalie_dist_to_goal",
    "goalie_lateral_speed",
    "shooter_goal_speed",
    "shooter_dist_to_goal",
    "shooter_angle_to_goal",
    "visible_angle",
    "shot_type"
]

pre_shot_features_print = [
    "Shot X", 
    "Shot Y", 
    "Total Defensive Pressure", 
    "# Defenders in Shooting Lane", 
    "# Defenders in Shadow Lane", 
    "# Pressures Left", 
    "# Pressures Right", 
    "# Pressures Front", 
    "# Pressures Back", 
    "Goalie Angle to Shooter", 
    "Goalie in Shooting Lane", 
    "Goalie in Shadow Lane", 
    "Goalie Distance to Goal", 
    "Goalie Speed", 
    "Goalie Lateral Speed", 
    "Shooter Speed", 
    "Shooter Lateral Speed", 
    "Shooter Goalwards Speed", 
    "Shooter Distance to Goal", 
    "Shooter Angle to Goal", 
    "Visible Angle",
    "Shot Type"
]

# Maintain order
pre_shot_features_pruned = [f for f in pre_shot_features if f not in ["num_pressures_left", "num_pressures_right"]]

pre_shot_features_speed = pre_shot_features_pruned + ["shot_speed"]

post_shot_features = [
    "shot_speed",
    "goalline_y_norm",
    "goalline_z",
    "on_goal",
    "dist_to_post",
    "dist_to_corner",
    "dist_to_center"
]

post_shot_features_full = pre_shot_features_pruned + post_shot_features

post_shot_features_minimal = pre_shot_features_minimal + post_shot_features

post_shot_features_one_hot = post_shot_features_full + [f"shot_type_{cat}" for cat in ShotType.categories]
post_shot_features_one_hot.remove("shot_type")  # Remove the original shot_type column since we have one-hot encoded columns

post_shot_features_xg = ["pre_shot"] + post_shot_features

feature_sets = {
    "pre_shot": pre_shot_features,
    "pre_shot_pruned": pre_shot_features_pruned,
    "pre_shot_minimal": pre_shot_features_minimal,
    "pre_shot_speed": pre_shot_features_speed,
    "post_shot_full": post_shot_features_full,
    "post_shot_minimal": post_shot_features_minimal,
    "post_shot_xg": post_shot_features_xg,
    "post_shot_one_hot": post_shot_features_one_hot
}

post_shot_feature_groups = {
    # shot origin
    'shot_origin': ["shot_x", "shot_y", "shooter_dist_to_goal", "shooter_angle_to_goal", "visible_angle"],
    # shot type
    'shot_type': ["shot_type_wristshot", "shot_type_slapshot", "shot_type_snapshot", "shot_type_backhand", 
        "shot_type_forehandbackhand", "shot_type_backhandforehand", "shot_type_wraparound", "shot_type_deflected"],
    # shooter_movement
    'shooter_movement': ["shooter_speed", "shooter_lateral_speed", "shooter_goal_speed"],
    # defender_positioning
    'defender_positioning': ["total_pressure", "num_defenders_in_shooting_lane", "num_defenders_in_shadow_lane", 
        "num_pressures_front", "num_pressures_back"],
    # goalie_positioning
    'goalie_positioning': ["goalie_angle_to_shooter", "goalie_in_shooting_lane", "goalie_in_shadow_lane", 
        "goalie_dist_to_goal", "goalie_speed", "goalie_lateral_speed"],
    # shot_execution
    'shot_execution': ["shot_speed", "goalline_y_norm", "goalline_z", "on_goal", 
        "dist_to_post", "dist_to_corner", "dist_to_center"]
}

def get_features(feature_set: str) -> list[str]:
    match feature_set:
        case "pre_shot":
            return pre_shot_features
        case "pre_shot_pruned":
            return pre_shot_features_pruned
        case "pre_shot_minimal":
            return pre_shot_features_minimal
        case "pre_shot_speed":
            return pre_shot_features_speed
        case "post_shot_full":
            return post_shot_features_full
        case "post_shot_minimal":
            return post_shot_features_minimal
        case "post_shot_xg":
            return post_shot_features_xg
        case "post_shot_one_hot":
            return post_shot_features_one_hot
        case _:
            raise ValueError(f"Unknown feature set: {feature_set}")