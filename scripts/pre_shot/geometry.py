import polars as pl
from polars import col as c

from utils import cross_product, _as_expr


def _point_in_convex_polygon(
    point_x_expr: pl.Expr | str,
    point_y_expr: pl.Expr | str,
    vertices: list[tuple[pl.Expr | str, pl.Expr | str]]
) -> pl.Expr:
    """
    Function to determine if a point is inside a convex polygon defined by a list of vertices.
    """
    point_x_expr = _as_expr(point_x_expr)
    point_y_expr = _as_expr(point_y_expr)

    normalized_vertices = [(_as_expr(x_expr), _as_expr(y_expr)) for x_expr, y_expr in vertices]
    edge_cross_products = []

    for index, (x1_expr, y1_expr) in enumerate(normalized_vertices):
        x2_expr, y2_expr = normalized_vertices[(index + 1) % len(normalized_vertices)]
        edge_cross_products.append(
            cross_product(
                x1_expr=x2_expr - x1_expr,
                y1_expr=y2_expr - y1_expr,
                x2_expr=point_x_expr - x1_expr,
                y2_expr=point_y_expr - y1_expr,
            )
        )

    return (
        pl.all_horizontal(*[cross_product_expr >= 0 for cross_product_expr in edge_cross_products])
        | pl.all_horizontal(*[cross_product_expr <= 0 for cross_product_expr in edge_cross_products])
    )


def angle_to_shooter(
    shooter_x_expr: pl.Expr | str,
    shooter_y_expr: pl.Expr | str,
    defender_x_expr: pl.Expr | str,
    defender_y_expr: pl.Expr | str
) -> pl.Expr:
    """
    Function to calculate the angle from a defender to the shooter relative to the goal, in degrees.
    Positive angles indicate the defender is to the right of the shooter (from shooter's perspective),
    while negative angles indicate the defender is to the left of the shooter.
    """
    GOAL_X = 89.0
    GOAL_Y = 0.0

    shooter_x_expr = _as_expr(shooter_x_expr)
    shooter_y_expr = _as_expr(shooter_y_expr)
    defender_x_expr = _as_expr(defender_x_expr)
    defender_y_expr = _as_expr(defender_y_expr)

    # Vector from shooter to goal
    x_sg = GOAL_X - shooter_x_expr
    y_sg = GOAL_Y - shooter_y_expr

    # Vector from shooter to defender
    x_sd = defender_x_expr - shooter_x_expr
    y_sd = defender_y_expr - shooter_y_expr

    angle_diff_rad = pl.arctan2(y_sg, x_sg) - pl.arctan2(y_sd, x_sd) # Right pressure positive, left pressure negative
    angle_diff_deg = ((angle_diff_rad.degrees() + 180) % 360) - 180  # Normalize to [-180, 180]
    
    return angle_diff_deg.alias("angle_to_shooter")