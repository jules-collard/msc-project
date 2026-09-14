import polars as pl
import numpy as np
import plotnine as gg
from plotnine import aes

def _circle_points(x, y, radius, start_angle=0, end_angle=2*np.pi, n=100):
    """Helper function to generate circle/arc coordinates."""
    t = np.linspace(start_angle, end_angle, n)
    return pl.DataFrame({'x': x + radius * np.cos(t), 'y': y + radius * np.sin(t)})

def geom_rink(fill=None):
    """
    Returns a list of plotnine geoms representing an NHL hockey rink.
    Coordinates: Center Ice at (0,0), X from -100 to 100, Y from -42.5 to 42.5.
    """
    layers = []
    
    # 2. Creases (Light Blue Semi-circles)
    # The calculated arc is part of a circle with radius 5, centered 1 ft behind the goal line.
    angle = np.arcsin(4/5) # 0.8 radians
    
    # Right Crease (Goal line at x=89, center of arc at x=88)
    crease_right_arc = _circle_points(88, 0, 5, np.pi - angle, np.pi + angle, 50)
    crease_right = pl.concat([
        crease_right_arc,
        pl.DataFrame({'x': [89., 89.], 'y': [-4., 4.]})
    ])
    layers.append(gg.geom_polygon(data=crease_right, mapping=aes(x='x', y='y'), 
                                  fill='lightblue', color='red', inherit_aes=False))
                                  
    # Left Crease (Goal line at x=-89, center of arc at x=-88)
    crease_left_arc = _circle_points(-88, 0, 5, angle, -angle, 50)
    crease_left = pl.concat([
        crease_left_arc,
        pl.DataFrame({'x': [-89., -89.], 'y': [-4., 4.]})
    ])
    layers.append(gg.geom_polygon(data=crease_left, mapping=aes(x='x', y='y'), 
                                  fill='lightblue', color='red', inherit_aes=False))

    # Goal Nets
    net_depth = 40 / 12  
    
    net_right = pl.DataFrame({
        'x': [89., 89. + net_depth, 89. + net_depth, 89.],
        'y': [3., 3., -3., -3.]
    })
    
    net_left = pl.DataFrame({
        'x': [-89., -89. - net_depth, -89. - net_depth, -89.],
        'y': [3., 3., -3., -3.]
    })
    
    layers.append(gg.geom_polygon(data=net_right, mapping=aes(x='x', y='y'), color='grey', fill='grey', size=1, inherit_aes=False))
    layers.append(gg.geom_polygon(data=net_left, mapping=aes(x='x', y='y'), color='grey', fill='grey', size=1, inherit_aes=False))
    
    # Static Lines (Goal lines, Blue lines, Center line)
    # The goal lines endpoints are calculated where X=89 intersects the 28ft corner radius (Y +/- 36.75)
    lines = pl.DataFrame({
        'x':    [89., -89., 25., -25., 0.],
        'xend': [89., -89., 25., -25., 0.],
        'y':    [-36.75, -36.75, -42.5, -42.5, -42.5],
        'yend': [36.75, 36.75, 42.5, 42.5, 42.5],
        'color': ['red', 'red', 'blue', 'blue', 'red'],
        'size':  [1., 1., 1.5, 1.5, 1.]
    })
    
    for row in lines.iter_rows(named=True):
        layers.append(gg.geom_segment(
            data=pl.DataFrame([row]), 
            mapping=aes(x='x', y='y', xend='xend', yend='yend'), 
            color=row['color'], size=row['size'], inherit_aes=False))
        
    # Center Circle
    layers.append(gg.geom_path(data=_circle_points(0, 0, 15), mapping=aes(x='x', y='y'), 
                            color='blue', size=1, inherit_aes=False))
    
    # End-zone Face-off circles
    for cx in [69, -69]:
        for cy in [22, -22]:
            layers.append(gg.geom_path(data=_circle_points(cx, cy, 15), mapping=aes(x='x', y='y'), 
                                    color='red', size=1, inherit_aes=False))
            
    # Face-off dots
    spots_red_centers = [(69, 22), (69, -22), (-69, 22), (-69, -22), (20, 22), (20, -22), (-20, 22), (-20, -22)]
    for cx, cy in spots_red_centers:
        layers.append(gg.geom_polygon(data=_circle_points(cx, cy, 1), mapping=aes(x='x', y='y'), 
                                   fill='red', color='red', inherit_aes=False))
        
    layers.append(gg.geom_polygon(data=_circle_points(0, 0, 0.5), mapping=aes(x='x', y='y'), 
                               fill='blue', color='blue', inherit_aes=False))

    # 7. Goalie Trapezoid Lines
    trapezoid = pl.DataFrame({
        'x':    [89., 89., -89., -89.],
        'xend': [100., 100., -100., -100.],
        'y':    [11., -11., 11., -11.],
        'yend': [14., -14., 14., -14.]
    })
    
    for row in trapezoid.iter_rows(named=True):
        layers.append(gg.geom_segment(
            data=pl.DataFrame([row]), 
            mapping=aes(x='x', y='y', xend='xend', yend='yend'), 
            color='red', size=1, inherit_aes=False))

    # Face-off Circle Hash Marks
    hash_marks_data = []
    # 5 feet 7 inches (67 inches) apart -> offset from center is 33.5 inches (2.7917 feet)
    dx = (67 / 12) / 2 
    # Calculate exactly where this intersects the 15-foot radius circle
    dy_start = np.sqrt(15**2 - dx**2)  
    dy_end = dy_start + 2  # extend outward by 2 feet
    
    for cx in [69, -69]:
        for cy in [22, -22]:
            for sign_x in [-1, 1]:      # Left and right hash mark in each pair
                for sign_y in [-1, 1]:  # Pair closest to boards (+1) and closest to center (-1)
                    x = cx + sign_x * dx
                    y_start = cy + sign_y * dy_start
                    y_end = cy + sign_y * dy_end
                    hash_marks_data.append({'x': x, 'xend': x, 'y': y_start, 'yend': y_end})
                    
    hash_marks = pl.DataFrame(hash_marks_data)
    
    for row in hash_marks.iter_rows(named=True):
        layers.append(gg.geom_segment(
            data=pl.DataFrame([row]),
            mapping=aes(x='x', y='y', xend='xend', yend='yend'),
            color='red', size=1, inherit_aes=False
        ))

    # Boards (Outer Rink perimeter)
    # The NHL corner radius is 28 feet. Centers are at +/- 72 X and +/- 14.5 Y
    t = np.linspace(0, np.pi/2, 25)
    schema = {'x': pl.Float32, 'y': pl.Float32}
    tr = pl.DataFrame({'x': 72 + 28*np.cos(t), 'y': 14.5 + 28*np.sin(t)}, schema=schema)
    tl = pl.DataFrame({'x': -72 + 28*np.cos(t + np.pi/2), 'y': 14.5 + 28*np.sin(t + np.pi/2)}, schema=schema)
    bl = pl.DataFrame({'x': -72 + 28*np.cos(t + np.pi), 'y': -14.5 + 28*np.sin(t + np.pi)}, schema=schema)
    br = pl.DataFrame({'x': 72 + 28*np.cos(t + 3*np.pi/2), 'y': -14.5 + 28*np.sin(t + 3*np.pi/2)}, schema=schema)
    boards = pl.concat([tr, tl, bl, br])
    
    layers.append(gg.geom_polygon(data=boards, mapping=aes(x='x', y='y'), 
                               fill=None, color='black', size=1, inherit_aes=False))

    # Ice background fill (optional)
    if fill:
        layers.insert(0, gg.geom_polygon(data=boards, mapping=aes(x='x', y='y'),
                                   fill=fill, color=None, inherit_aes=False))

    return layers

def geom_net():
    layers = []

    net_outline = pl.DataFrame({
        'y': [-3, -3, 3, 3],
        'z': [0, 4, 4, 0]
    })

    # Net
    layers.append(gg.geom_rect(
        aes(xmin=-3, xmax=3, ymin=0, ymax=4, alpha=0.8), fill='lightgrey', show_legend=False
    ))

    # Posts
    layers.append(gg.geom_path(
        aes(x='y', y='z'), 
        data=net_outline, 
        color="red", size=2, lineend="round"
    ))

    return layers

def geom_ice():
    return gg.geom_segment(
        aes(x=-5, xend=5, y=0, yend=0), 
        color="lightblue", size=2,
        inherit_aes=False
    )
