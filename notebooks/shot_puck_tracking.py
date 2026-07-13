import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    from polars import col as c
    from polars import selectors as cs
    import plotnine as p9
    from plotnine import ggplot, aes, geom_line, geom_vline, geom_hline
    from hockey_rink import NHLRink
    from matplotlib import pyplot as plt

    from data_readers import read_events, read_puck_tracking, read_entity_tracking
    from tracking_processing import derive_game_clock, adjust_vectors, calculate_goal_vectors, calculate_magnitudes
    from event_processing import add_flip
    from features.puck import calculate_shot_features
    from utils import distance_2d


@app.cell
def _():
    events = read_events("data/one_game/NHL_20252026_playoffs_20260521_MTLvsCAR_sapifullevents.json").pipe(add_flip)
    shots = events.filter(c('name') == 'shot').sort(c('game_time')).with_row_index(name='shot_id')
    return (shots,)


@app.cell
def _():
    periods = [1,2,3]
    player_tracking = read_entity_tracking(
        "data/one_game/NHL_20252026_postseason_20260521_MTLvsCAR_entity_tracking_processed_measurements.parquet"
    )

    puck_tracking = read_puck_tracking(
        [f"data/one_game/HOCKEY_NHL_2026_05_21_MTL@CAR_HITS311_Period_{i}.json" for i in periods], periods
    )

    puck_tracking = (
        pl.concat([player_tracking, puck_tracking], how='diagonal_relaxed')
        .pipe(derive_game_clock)
        .filter(c('clock_state') == 1, c('entity_id') == '1')
        .sort(c('game_time'))
        .drop(c('entity_id', 'entity_official_id', 'segment_idx', 'clock_state', 'raw_x', 'raw_y', 'raw_z'))
    )
    return (puck_tracking,)


@app.cell
def _(puck_tracking, shots):
    WINDOW_SIZE = 1.6

    shot_info = calculate_shot_features(shots, puck_tracking, WINDOW_SIZE).collect()

    tracking_with_shots = (
        puck_tracking.sort(c('game_time'))
        .join_asof(
            shots.sort(c('game_time')),
            on='game_time',
            strategy='nearest',
            tolerance=WINDOW_SIZE,
            coalesce=False
        ).drop_nulls(c('shot_id'))
        .pipe(adjust_vectors)
        .collect()
        .pipe(calculate_goal_vectors)
        .pipe(calculate_magnitudes)
        .with_columns(
            dist_to_shot = distance_2d('x_adj', 'y_adj', 'x_adj_coord', 'y_adj_coord').alias('dist_to_shot')
        ).with_columns(
            angle_acc = c('angle_to_goal').diff().over(c('shot_id'))
        )
    )
    return shot_info, tracking_with_shots


@app.cell
def _(shot_info, shots):
    max_id = shots.select(c('shot_id').max()).collect().item()

    id_selector = mo.ui.dropdown.from_series(shot_info.select(c('shot_id').unique()).to_series(), allow_select_none=False, label="Select Shot ID", value=0)
    return (id_selector,)


@app.cell
def _(id_selector, shot_info, shots, tracking_with_shots):
    sample = tracking_with_shots.filter(c('shot_id') == id_selector.value)
    game_time = shots.filter(c('shot_id') == id_selector.value).select(c('game_time')).collect().item()
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


@app.cell
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
        + geom_line(aes(x='game_time', y='speed'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Speed (ft/s)')
        + custom_theme
    )

    goal_speed_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='game_time', y='goal_speed'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Velocity Towards Goal (ft/s)')
        + custom_theme
    )

    acc_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='game_time', y='acceleration'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Acceleration (ft/s²)')
        + custom_theme
    )

    goal_acc_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='game_time', y='goal_acceleration'))
        + geom_vline(xintercept=game_time, color='red')
        + geom_hline(yintercept=0, color='grey')
        + p9.labs(x='Game Time (s)', y='Acc. Towards Goal (ft/s²)')
        + custom_theme
    )

    angle_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='game_time', y='angle_to_goal'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.scale_y_continuous(limits=(-180, 180), breaks=[-180, -90, 0, 90, 180])
        + p9.labs(x='Game Time (s)', y='Angle to Goal (degrees)')
        + custom_theme
    )

    angle_acc_plot = (
        ggplot(sample)
        + p9.geom_rect(xmin=shot_time, xmax=shot_end_time, ymin=-float('Inf'), ymax=float('Inf'), alpha=0.01)
        + geom_line(aes(x='game_time', y='angle_acc'))
        + geom_vline(xintercept=game_time, color='red')
        + p9.labs(x='Game Time (s)', y='Angular Acceleration (degrees/s)')
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

        angle_acc_plot += shot_time_marker
    return (
        acc_plot,
        angle_acc_plot,
        angle_plot,
        custom_theme,
        goal_acc_plot,
        goal_speed_plot,
        speed_plot,
    )


@app.cell
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


@app.cell
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
        + custom_theme
        + p9.theme(title=p9.element_blank(), axis_title=p9.element_blank())
    )
    return (shot_plot,)


@app.cell
def _(shots):
    shots.collect()
    return


@app.cell
def _(
    acc_plot,
    angle_acc_plot,
    angle_plot,
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
        (speed_plot | acc_plot | angle_plot) / (goal_speed_plot | goal_acc_plot | angle_acc_plot)
    ])
    return


if __name__ == "__main__":
    app.run()
