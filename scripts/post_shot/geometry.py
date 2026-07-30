import polars as pl
from polars import col as c

from utils import distance_to_point_2d, magnitude_2d

def goal_vectors(
    x_expr: pl.Expr | str = 'x_adj',
    y_expr: pl.Expr | str = 'y_adj',
    vx_expr: pl.Expr | str = 'vx_adj',
    vy_expr: pl.Expr | str = 'vy_adj',
    ax_expr: pl.Expr | str = 'ax_adj',
    ay_expr: pl.Expr | str = 'ay_adj'
) -> list[pl.Expr]:
    """
    Function to calculate goal vector features (goal_speed, goal_acceleration, angle_to_goal) from input expressions.
    Projects velocity and acceleration onto the vector from the puck to the goal, and calculates the angle of the velocity
    vector relative to the goal vector.
    """

    x_expr = c(x_expr) if isinstance(x_expr, str) else x_expr
    y_expr = c(y_expr) if isinstance(y_expr, str) else y_expr
    vx_expr = c(vx_expr) if isinstance(vx_expr, str) else vx_expr
    vy_expr = c(vy_expr) if isinstance(vy_expr, str) else vy_expr
    ax_expr = c(ax_expr) if isinstance(ax_expr, str) else ax_expr
    ay_expr = c(ay_expr) if isinstance(ay_expr, str) else ay_expr

    GOAL_X = 89
    GOAL_Y = 0

    # Vector to goal and distance
    dx = GOAL_X - x_expr
    dy = GOAL_Y - y_expr
    dist = distance_to_point_2d(x_expr, y_expr, GOAL_X, GOAL_Y) + 1e-6
    
    # Unit vector components
    u_x = dx / dist
    u_y = dy / dist

    goal_speed = ((vx_expr * u_x) + (vy_expr * u_y))
    goal_acceleration = ((ax_expr * u_x) + (ay_expr * u_y))
    tangent_speed = ((u_x * vy_expr) - (u_y * vx_expr))
    
    return [
        goal_speed.alias("goal_speed"),
        goal_acceleration.alias("goal_acceleration"),
        pl.arctan2(tangent_speed, goal_speed).degrees().alias("angle_to_goal")
    ]

def project_y_to_goalline(x1_col: str, y1_col: str, x2_col: str, y2_col: str) -> pl.Expr:
    """Returns a polars expression to project a vector onto the goal line, returning the y-coordinate of the projection."""

    GOAL_X = 89.0

    delta_x = c(x2_col) - c(x1_col)
    delta_y = c(y2_col) - c(y1_col)

    return (
        pl.when(delta_x > 0)
        .then(
            c(y1_col) + (delta_y / delta_x) * (GOAL_X - c(x1_col))
        ).otherwise(None)
        .clip(-42.5, 42.5) # Clip to avoid obscenely large projections for shots almost parallel to the goal line
        .alias('goalline_y')
    )

def project_z_to_goalline(
    x1: str, z1: str, t1: str, 
    x2: str, z2: str, t2: str, 
    gravity: float = 32.174 # Standard gravity in ft/s^2
) -> pl.Expr:
    """Returns a polars expression projecting puck height from the point of max speed."""
    
    GOAL_X = 89.0
    
    delta_x = c(x2) - c(x1)
    delta_z = c(z2) - c(z1)
    dt = c(t2) - c(t1)
    
    # 1. Remaining time to travel from x2 to the goal line
    t_remaining = dt * (GOAL_X - c(x2)) / delta_x
    
    # 2. Vertical velocity at (x2, z2)
    v_z2 = (delta_z / dt) - (0.5 * gravity * dt)
    
    # 3. Projected height at the goal line starting from z2
    raw_z_goal = pl.col(z2) + (v_z2 * t_remaining) - (0.5 * gravity * t_remaining**2)
    
    return (
        pl.when((delta_x > 0) & (dt > 0))
        .then(
            pl.max_horizontal(0.0, raw_z_goal)
        ).otherwise(None)
        .alias('goalline_z')
    )