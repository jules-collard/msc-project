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
    "Visible Angle"
]

# Maintain order
pre_shot_features_pruned = [f for f in pre_shot_features if f not in ["num_pressures_left", "num_pressures_right"]]

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

post_shot_features_minimal = post_shot_features + ['pre_shot']