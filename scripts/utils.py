import polars as pl
from polars import col as c

def distance_2d(x1_col: str, y1_col: str, x2_col: str, y2_col: str) -> pl.Expr:
    """Returns a Polars expression to calculate 2D Euclidean distance."""
    return (
        (c(x1_col) - c(x2_col)).pow(2) + 
        (c(y1_col) - c(y2_col)).pow(2)
    ).sqrt()

def distance_to_point_2d(x_col: str, y_col: str, point_x: float, point_y: float) -> pl.Expr:
    """Returns a Polars expression to calculate 2D Euclidean distance to a fixed point."""
    return (
        (c(x_col) - point_x).pow(2) + 
        (c(y_col) - point_y).pow(2)
    ).sqrt()

def magnitude_2d(vx_col: str, vy_col: str) -> pl.Expr:
    """Returns a Polars expression to calculate the magnitude of a 2D vector."""
    return (c(vx_col).pow(2) + c(vy_col).pow(2)).sqrt()

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