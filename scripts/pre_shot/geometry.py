import polars as pl
from polars import col as c

from utils import cross_product, _as_expr, magnitude_2d, distance_to_point_2d


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

def project_vector(
    x_vec_expr: pl.Expr | str,
    y_vec_expr: pl.Expr | str,
    x_start_expr: pl.Expr | str,
    y_start_expr: pl.Expr | str,
    x_end_expr: pl.Expr | str,
    y_end_expr: pl.Expr | str
) -> pl.Expr:
    """
    Returns a polars expression to project a vector (x_vec, y_vec) onto the vector defined by (x_start, y_start) -> (x_end, y_end).
    The result is the scalar projection of the vector onto the goal vector.
    """

    x_vec_expr = c(x_vec_expr) if isinstance(x_vec_expr, str) else x_vec_expr
    y_vec_expr = c(y_vec_expr) if isinstance(y_vec_expr, str) else y_vec_expr
    x_start_expr = c(x_start_expr) if isinstance(x_start_expr, str) else x_start_expr
    y_start_expr = c(y_start_expr) if isinstance(y_start_expr, str) else y_start_expr
    x_end_expr = c(x_end_expr) if isinstance(x_end_expr, str) else x_end_expr
    y_end_expr = c(y_end_expr) if isinstance(y_end_expr, str) else y_end_expr

    # Vector to project onto and distance
    dx = x_end_expr - x_start_expr
    dy = y_end_expr - y_start_expr
    dist = magnitude_2d(dx, dy) + 1e-6  # Add small epsilon to avoid division by zero
    
    # Unit vector components
    u_x = dx / dist
    u_y = dy / dist

    projection = ((x_vec_expr * u_x) + (y_vec_expr * u_y))
    
    return projection.alias('projection')

def visible_angle(x_expr: pl.Expr | str, y_expr: pl.Expr | str) -> pl.Expr:
    """
    Returns a polars expression to calculate the visible angle from a point (x, y) to the goal,
    where the angle is the angle subtended by the goal posts at the point (x, y).
    """
    x_expr = _as_expr(x_expr)
    y_expr = _as_expr(y_expr)

    GOAL_X = 89.0
    GOAL_Y_1 = 3.0
    GOAL_Y_2 = -3.0

    d_1 = distance_to_point_2d(x_expr, y_expr, GOAL_X, GOAL_Y_1)
    d_2 = distance_to_point_2d(x_expr, y_expr, GOAL_X, GOAL_Y_2)
    ratio = (d_1.pow(2) + d_2.pow(2) - 6.0 ** 2) / (2 * d_1 * d_2 + 1e-6)  # Add small epsilon to avoid division by zero
    angle = ratio.arccos().degrees() # Law of cosines
    
    return angle.alias('visible_angle')