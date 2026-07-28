import polars as pl

from pre_shot.geometry import _point_in_convex_polygon
from utils import _as_expr


def inside_shooting_lane(
    shooter_x_expr: pl.Expr | str,
    shooter_y_expr: pl.Expr | str,
    defender_x_expr: pl.Expr | str,
    defender_y_expr: pl.Expr | str
) -> pl.Expr:
    """
    Function to determine if a defender is inside the shooting lane of a shooter.
    The shooting lane is defined by the line segment between the two goal posts and the shooter.
    Returns a boolean expression indicating whether the defender is inside the shooting lane.
    """
    GOAL_X = 89.0
    GOAL_Y_1 = 3.0
    GOAL_Y_2 = -3.0

    shooter_x_expr = _as_expr(shooter_x_expr)
    shooter_y_expr = _as_expr(shooter_y_expr)
    defender_x_expr = _as_expr(defender_x_expr)
    defender_y_expr = _as_expr(defender_y_expr)

    return _point_in_convex_polygon(
        defender_x_expr,
        defender_y_expr,
        [
            (shooter_x_expr, shooter_y_expr),
            (GOAL_X, GOAL_Y_1),
            (GOAL_X, GOAL_Y_2),
        ],
    ).alias("inside_shooting_lane")

def inside_shadow_lane(
    shooter_x_expr: pl.Expr | str,
    shooter_y_expr: pl.Expr | str,
    defender_x_expr: pl.Expr | str,
    defender_y_expr: pl.Expr | str,
    lane_expansion: float = 3.0
) -> pl.Expr:
    """
    Determines if a defender is inside the 'shadow lane' (a widened shooting lane).
    Widens the goalposts by `lane_expansion` and creates a perpendicular line 
    at the shooter of width `lane_expansion * 2`.
    """
    GOAL_X = 89.0
    G1_Y_WIDENED = 3.0 + lane_expansion   # Left post widened
    G2_Y_WIDENED = -3.0 - lane_expansion  # Right post widened

    shooter_x_expr = _as_expr(shooter_x_expr)
    shooter_y_expr = _as_expr(shooter_y_expr)
    defender_x_expr = _as_expr(defender_x_expr)
    defender_y_expr = _as_expr(defender_y_expr)

    dx = GOAL_X - shooter_x_expr
    dy = -shooter_y_expr
    dist = (dx.pow(2) + dy.pow(2)).sqrt()

    perp_x = -dy / dist
    perp_y = dx / dist

    sl_x = shooter_x_expr + (lane_expansion * perp_x)
    sl_y = shooter_y_expr + (lane_expansion * perp_y)

    sr_x = shooter_x_expr - (lane_expansion * perp_x)
    sr_y = shooter_y_expr - (lane_expansion * perp_y)

    return _point_in_convex_polygon(
        defender_x_expr,
        defender_y_expr,
        [
            (sl_x, sl_y),
            (GOAL_X, G1_Y_WIDENED),
            (GOAL_X, G2_Y_WIDENED),
            (sr_x, sr_y),
        ],
    ).alias("inside_shadow_lane")