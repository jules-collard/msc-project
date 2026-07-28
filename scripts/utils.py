import polars as pl
from polars import col as c


def _as_expr(value: pl.Expr | str) -> pl.Expr:
    return c(value) if isinstance(value, str) else value

def distance_2d(
    x1_expr: pl.Expr | str,
    y1_expr: pl.Expr | str,
    x2_expr: pl.Expr | str,
    y2_expr: pl.Expr | str
) -> pl.Expr:
    """Returns a Polars expression to calculate 2D Euclidean distance."""
    x1_expr = c(x1_expr) if isinstance(x1_expr, str) else x1_expr
    y1_expr = c(y1_expr) if isinstance(y1_expr, str) else y1_expr
    x2_expr = c(x2_expr) if isinstance(x2_expr, str) else x2_expr
    y2_expr = c(y2_expr) if isinstance(y2_expr, str) else y2_expr

    return (
        (x1_expr - x2_expr).pow(2) + 
        (y1_expr - y2_expr).pow(2)
    ).sqrt()

def distance_to_point_2d(
    x_expr: pl.Expr | str,
    y_expr: pl.Expr | str,
    point_x: float,
    point_y: float
) -> pl.Expr:
    """Returns a Polars expression to calculate 2D Euclidean distance to a fixed point."""
    x_expr = c(x_expr) if isinstance(x_expr, str) else x_expr
    y_expr = c(y_expr) if isinstance(y_expr, str) else y_expr

    return (
        (x_expr - point_x).pow(2) + 
        (y_expr - point_y).pow(2)
    ).sqrt()

def cross_product(
    x1_expr: pl.Expr | str,
    y1_expr: pl.Expr | str,
    x2_expr: pl.Expr | str,
    y2_expr: pl.Expr | str
) -> pl.Expr:
    """
    Function to calculate the cross product of two 2D vectors.
    The cross product is a scalar value that indicates the relative orientation of the two vectors.
    A positive value indicates that the second vector is counter-clockwise from the first vector,
    while a negative value indicates that it is clockwise.
    """

    x1_expr = c(x1_expr) if isinstance(x1_expr, str) else x1_expr
    y1_expr = c(y1_expr) if isinstance(y1_expr, str) else y1_expr
    x2_expr = c(x2_expr) if isinstance(x2_expr, str) else x2_expr
    y2_expr = c(y2_expr) if isinstance(y2_expr, str) else y2_expr

    return (x1_expr * y2_expr - y1_expr * x2_expr).alias("cross_product")

def magnitude_2d(vx_expr: pl.Expr | str, vy_expr: pl.Expr | str) -> pl.Expr:
    """Returns a Polars expression to calculate the magnitude of a 2D vector."""
    vx_expr = c(vx_expr) if isinstance(vx_expr, str) else vx_expr
    vy_expr = c(vy_expr) if isinstance(vy_expr, str) else vy_expr

    return (vx_expr.pow(2) + vy_expr.pow(2)).sqrt()

def cohens_kappa(tp: str, tn: str, fp: str, fn: str) -> pl.Expr:
    """Returns a Polars expression to calculate Cohen's Kappa statistic."""
    return (
        2 * (c(tp) * c(tn) - c(fn) * c(fp))
    ) / (
        (c(tp) + c(fp)) * (c(fp) + c(tn)) + (c(tp) + c(fn)) * (c(fn) + c(tn))
    )