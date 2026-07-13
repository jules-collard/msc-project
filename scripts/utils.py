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

def project_to_goalline(x1_col: str, y1_col: str, x2_col: str, y2_col: str) -> pl.Expr:
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