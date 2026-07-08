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