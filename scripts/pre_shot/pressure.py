import polars as pl
from polars import col as c

from utils import _as_expr


def pressure(
    angle_expr: pl.Expr | str,
    dist_expr: pl.Expr | str,
    d_front: float = 18,
    d_back: float = 5,
    q: float = 0.8
) -> pl.Expr:
    """
    Function to calculate the pressure exerted by a defender on a shooter based on the angle and distance to the shooter.
    See Andrienko et al. (2017) for details on the pressure model.

    Arguments:
        d_front: The maximum distance at which a defender can exert pressure on the shooter (in feet).
        d_back: The minimum distance at which a defender can exert pressure on the shooter (in feet).
        q: Exponent to regulate the speed of the distance decay,
    
    """
    
    angle_expr = c(angle_expr) if isinstance(angle_expr, str) else angle_expr
    dist_expr = c(dist_expr) if isinstance(dist_expr, str) else dist_expr

    z = (1 - angle_expr.radians().cos()) / 2
    L = d_front + (d_back - d_front) * (z.pow(3) + 0.3 * z) / 1.3
    return (1 - dist_expr / L).clip(0,1).pow(q).alias("pressure")


def pressure_direction(angle_expr: pl.Expr | str) -> pl.Expr:
    angle_expr = _as_expr(angle_expr)

    return (
        pl.when(angle_expr.is_between(-135, -45, closed='left')).then(pl.lit('left'))
        .when(angle_expr.is_between(-45, 45)).then(pl.lit('front'))
        .when(angle_expr.is_between(45, 135, closed='right')).then(pl.lit('right'))
        .otherwise(pl.lit('back'))
        .alias('pressure_direction')
    )