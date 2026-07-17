import polars as pl
from polars import col as c

from utils import distance_to_point_2d

def shot_features() -> pl.Expr:
    """
    Returns a list of expressions with post-shot features.
    """

    on_goal = (c('goalline_y').abs() <= 3) & (c('goalline_z') <= 4)
    dist_to_post = 3 - c('goalline_y').abs()
    dist_to_crossbar = 4 - c('goalline_z')
    dist_to_top_corner = distance_to_point_2d(c('goalline_y').abs(), c('goalline_z'), 3, 4)
    dist_to_top_corner = ( # Distance to top corner is negative when not on target
        pl.when(on_goal)
        .then(dist_to_top_corner)
        .otherwise(-dist_to_top_corner)
    )
    dist_to_center = distance_to_point_2d(c('goalline_y'), c('goalline_z'), 0, 2)
    nearest_post_y = pl.when(c('goalline_y') < 0).then(-3).otherwise(3)

    return [
        on_goal.alias('on_goal'),
        dist_to_post.alias('dist_to_post'),
        dist_to_crossbar.alias('dist_to_crossbar'),
        dist_to_top_corner.alias('dist_to_top_corner'),
        dist_to_center.alias('dist_to_center'),
        nearest_post_y.alias('nearest_post_y')
    ]