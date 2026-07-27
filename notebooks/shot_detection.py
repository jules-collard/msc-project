import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    import plotnine as p9
    from plotnine import ggplot, aes, geom_line, geom_vline, geom_hline
    from hockey_rink import NHLRink
    from matplotlib import pyplot as plt

    from data_readers import batch_read_puck_tracking, batch_read_events, batch_read_rosters
    from processing.tracking import adjust_vectors
    from post_shot.geometry import goal_vectors
    from post_shot.features import PostShotData
    from utils import distance_2d, magnitude_2d
    from interaction import game_selectors, display_game_selectors


@app.cell
def _():
    game_id_mapping = (
        pl.read_csv("mappings/NHL_20252026_game_smt_sportlogiq_id_map.csv")
        .with_columns(c('GameDate').str.strptime(pl.Date, format="%Y-%m-%d"))
    )
    return (game_id_mapping,)


@app.cell
def _(game_id_mapping):
    game_ids = game_id_mapping.select(c('SportlogiqGameID')).to_series()
    game_id_selector = mo.ui.dropdown.from_series(game_ids, value=game_ids.first())
    game_id_selector
    return (game_id_selector,)


@app.cell
def _(game_id_mapping, game_id_selector):
    games = (
        game_id_mapping
        .filter(
            c('SportlogiqGameID') == game_id_selector.value,
        )
    )

    sportlogiq_ids = games.select(c('SportlogiqGameID')).to_series().to_list()
    SMT_ids = games.select(c('SMTGameID')).to_series().to_list()
    return SMT_ids, games, sportlogiq_ids


@app.cell
def _(SMT_ids, games, sportlogiq_ids):
    events = batch_read_events([f"/data/sportlogiq/*/games/{id}/*_sapifullevents.json" for id in sportlogiq_ids])

    puck_tracking = batch_read_puck_tracking(
        [f"/data/smtoasis/*/games/{id}/*_puck_tracking_raw_measurements*.parquet" for id in SMT_ids],
        mapping=games.lazy()
    )

    player_info = batch_read_rosters([f"/data/sportlogiq/*/games/{id}/NHL_*_gameroster.json" for id in sportlogiq_ids])

    post_shot_data = PostShotData(events, puck_tracking, player_info)
    return (post_shot_data,)


@app.cell
def _(post_shot_data):
    WINDOW_SIZE = 1.6

    shot_info = post_shot_data.post_shot_features().collect()

    tracking_with_shots = (
        post_shot_data.puck_tracking_prepared
        .sort(c('game_id', 'period', 'elapsed_time'))
        .join_asof(
            post_shot_data.shots.sort(c('game_id', 'period', 'elapsed_time')),
            by=['game_id', 'period'],
            on='elapsed_time',
            strategy='nearest',
            tolerance=WINDOW_SIZE / 2,
            coalesce=False
        )
        .drop_nulls(c('shot_id'))
        .with_columns(adjust_vectors(c('x', 'y', 'vx', 'vy', 'ax', 'ay')))
        .with_columns(
            goal_vectors(),
            speed = magnitude_2d('vx', 'vy'),
            acceleration = magnitude_2d('ax', 'ay'),
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord')
        ).with_columns(
            angle_vel = c('angle_to_goal').diff().over(c('shot_id'))
        ).collect()
    )
    return shot_info, tracking_with_shots


@app.cell
def _(shot_info):
    ids = shot_info.filter().select(c('shot_id').unique()).to_series()
    first = ids.first()

    id_selector = mo.ui.dropdown.from_series(ids, allow_select_none=False, label="Shot #", value=first)
    return (id_selector,)


@app.cell
def _(id_selector, post_shot_data, shot_info, tracking_with_shots):
    sample = tracking_with_shots.filter(c('shot_id') == id_selector.value)
    game_time = post_shot_data.shots.filter(c('shot_id') == id_selector.value).select(c('elapsed_time')).collect().item()
    shot_logic = shot_info.filter(c('shot_id') == id_selector.value)

    shot_time = shot_logic.select(c('shot_time')).item()
    shot_end_time = shot_logic.select(c('shot_end_time')).item()
    shot_speed = shot_logic.select(c('shot_speed')).item()
    speed_time = shot_logic.select(c('traj_time')).item()
    return (
        game_time,
        sample,
        shot_end_time,
        shot_logic,
        shot_speed,
        shot_time,
        speed_time,
    )


@app.cell(hide_code=True)
def _(game_time, sample, shot_end_time, shot_speed, shot_time, speed_time):
    custom_theme = (
        p9.theme_bw(base_size=8)
        + p9.theme(
            axis_title_x=p9.element_blank(),
            axis_text_x=p9.element_text(angle=-90)
        )
    )

    logic_exists = all([shot_time is not None, shot_speed is not None, speed_time is not None])

    speed_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='speed'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Speed (ft/s)')
        + custom_theme
    )

    goal_speed_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='goal_speed'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Velocity Towards Goal (ft/s)')
        + custom_theme
    )

    acc_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='acceleration'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Acceleration (ft/s²)')
        + custom_theme
    )

    goal_acc_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='goal_acceleration'))
        + geom_vline(xintercept=game_time, color='red')
        + geom_hline(yintercept=0, color='grey')
        + p9.labs(x='Game Time (s)', y='Acc. Towards Goal (ft/s²)')
        + custom_theme
    )

    angle_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='angle_to_goal'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.scale_y_continuous(limits=(-180, 180), breaks=[-180, -90, 0, 90, 180])
        + p9.labs(x='Game Time (s)', y='Angle to Goal (degrees)')
        + custom_theme
    )

    angle_vel_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='elapsed_time', y='angle_vel'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Angular Velocity (degrees/s)')
        + custom_theme
    )

    if logic_exists:
        shot_time_marker = geom_vline(xintercept=shot_time, linetype='dashed', color='blue')

        speed_plot += shot_time_marker
        speed_plot += p9.geom_hline(aes(yintercept=shot_speed), color='blue', linetype='dotted')

        goal_speed_plot += shot_time_marker

        acc_plot += shot_time_marker

        goal_acc_plot += shot_time_marker

        angle_plot += shot_time_marker

        angle_vel_plot += shot_time_marker
    return (
        acc_plot,
        angle_plot,
        angle_vel_plot,
        custom_theme,
        goal_acc_plot,
        goal_speed_plot,
        speed_plot,
    )


@app.cell(hide_code=True)
def _(sample, shot_logic):
    fig, ax = plt.subplots(figsize=(8, 4))

    rink = NHLRink()

    rink.scatter(
        sample.select(c('x_adj')),
        sample.select(c('y_adj')),
        s=10,
        alpha=0.5,
        draw_kw={'display_range': 'ozone', 'rotation': 90}
    )

    rink.scatter(
        shot_logic.select(c('shot_x')),
        shot_logic.select(c('shot_y')),
        s=50,
        c='blue',
        draw_kw={'display_range': 'ozone', 'rotation': 90}
    )

    rink.scatter(
        shot_logic.select(c('traj_x')),
        shot_logic.select(c('traj_y')),
        s=50,
        c='blue',
        marker='x',
        draw_kw={'display_range': 'ozone', 'rotation': 90}
    )

    rink.scatter(
        sample.select(c('x_adj_coord').first()),
        sample.select(c('y_adj_coord').first()),
        s=50,
        c='red',
        draw_kw={'display_range': 'ozone', 'rotation': 90}
    )

    if shot_logic.select(c('goalline_y').is_null().not_()).item():
        rink.arrow(
            shot_logic.select(c('shot_x')).item(),
            shot_logic.select(c('shot_y')).item(),
            shot_logic.select(89 - c('shot_x')).item(),
            shot_logic.select(c('goalline_y') - c('shot_y')).item(),
            color='blue',
            width=0.005,
            head_width=0.5,
            draw_kw={'display_range': 'ozone', 'rotation': 90}
        )

    None
    return ax, rink


@app.cell(hide_code=True)
def _(custom_theme, shot_logic):
    # 1. Define the NHL Net Outline (in feet)
    # The path goes: Bottom-Left Post -> Top-Left -> Top-Right -> Bottom-Right
    net_outline = pl.DataFrame({
        'y': [-3, -3, 3, 3],
        'z': [0, 4, 4, 0]
    })

    # 3. Build the Plotnine Graphic
    shot_plot = (
        ggplot()
        + p9.coord_fixed(ratio=1, ylim=(0, 6))
        + p9.scale_x_reverse(limits=(5,-5))  # Flip the x-axis to match hockey rink orientation
        # Net
        + p9.geom_rect(
            aes(xmin=-3, xmax=3, ymin=0, ymax=4), fill='lightgrey',
        )
        # Ice Surface
        + p9.geom_segment(
            aes(x=-5, xend=5, y=0, yend=0), 
            color="lightblue", size=2
        ) 
        # Posts
        + p9.geom_path(
            aes(x='y', y='z'), 
            data=net_outline, 
            color="red", size=2, lineend="round"
        )   
        # Shots
        + p9.geom_point(
            aes(x='goalline_y', y='goalline_z'), size=5,
            data=shot_logic,
        )
        # Feature Illustrations
        + p9.geom_segment(
            # Distance to Nearest Top Corner
            aes(x='goalline_y', xend='nearest_post_y', y='goalline_z', yend=4),
            data=shot_logic,
            linetype="dotted"
        ) + p9.geom_segment(
            # Distance to Nearest Post
            aes(x='goalline_y', xend='nearest_post_y', y='goalline_z', yend='goalline_z'),
            data=shot_logic,
            linetype="dotted"
        )+ p9.geom_segment(
            # Distance to Center
            aes(x='goalline_y', xend=0, y='goalline_z', yend=2),
            data=shot_logic,
            linetype="dotted"
        ) + p9.geom_text(
            aes(x='goalline_y', y='goalline_z', label='shot_speed'),
            data=shot_logic,
            nudge_y=-0.5,
            size=8,
            color='blue',
            format_string='{:.1f}ft/s'
        ) + p9.geom_text(
            aes(x='nearest_post_y', y=4, label='dist_to_top_corner'),
            data=shot_logic,
            nudge_y=0.2,
            size=8,
            color='blue',
            format_string='{:.1f}ft'
        ) + p9.geom_text(
            aes(x='nearest_post_y', y='goalline_z', label='dist_to_post'),
            data=shot_logic,
            nudge_x=0.4,
            size=8,
            color='blue',
            format_string='{:.1f}ft'
        ) + p9.geom_text(
            aes(x=0, y=2, label='dist_to_center'),
            data=shot_logic,
            nudge_y=0.2,
            size=8,
            color='blue',
            format_string='{:.1f}ft'
        )
        + custom_theme
        + p9.theme(title=p9.element_blank(), axis_title=p9.element_blank())
    )
    return (shot_plot,)


@app.cell
def _(
    acc_plot,
    angle_plot,
    angle_vel_plot,
    ax,
    goal_acc_plot,
    goal_speed_plot,
    id_selector,
    rink,
    shot_plot,
    speed_plot,
):
    mo.vstack([
        id_selector,
        mo.hstack([shot_plot, rink.draw(ax=ax, display_range="ozone", rotation=90)], widths="equal"),
        (speed_plot | acc_plot | angle_plot) / (goal_speed_plot | goal_acc_plot | angle_vel_plot)
    ])
    return


if __name__ == "__main__":
    app.run()
