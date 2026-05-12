import numpy as np
import plotly.graph_objects as go


def add_box(fig, center, half_extents, color='lightblue', opacity=0.4, name='box'):
    cx, cy, cz = center
    hx, hy, hz = half_extents

    xs = [cx + sx * hx for sx in [-1, 1, 1, -1, -1, 1, 1, -1]]
    ys = [cy + sy * hy for sy in [-1, -1, -1, -1,  1, 1,  1,  1]]
    zs = [cz + sz * hz for sz in [-1, -1,  1,  1, -1,-1,  1,  1]]

    i = [0,0, 1,1, 0,0, 4,4, 0,0, 3,3]
    j = [1,2, 2,3, 4,5, 5,6, 1,5, 2,6]
    k = [2,3, 3,0, 5,6, 6,7, 5,4, 6,7]

    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=True,
    ))


def add_frame(fig, origin, R=np.eye(3), axis_len=0.05, name='frame'):
    for label, d, color in [('x', [1,0,0], 'red'), ('y', [0,1,0], 'green'), ('z', [0,0,1], 'blue')]:
        tip = origin + R @ np.array(d) * axis_len
        fig.add_trace(go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
            mode='lines', line=dict(color=color, width=6),
            name=f'{name}:{label}', showlegend=False,
        ))


def add_joint(fig, pos, name='joint'):
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers+text',
        marker=dict(color='orange', size=8, symbol='diamond'),
        text=[name], textposition='middle right',
        name=name,
    ))


if __name__ == '__main__':
    # from XML — all values in world frame at q=0
    #   link_0: body origin (box center) at [0, 0.15, 0],  joint pivot at [0, 0,    0]
    #   link_1: body origin (box center) at [0, 0.45, 0],  joint pivot at [0, 0.30, 0]

    link0_origin = np.array([0.0, 0.15, 0.0])
    link1_origin = np.array([0.0, 0.45, 0.0])
    half_extents = np.array([0.025, 0.15, 0.025])

    # joint pivots = link frame origins per slides convention
    # (frame origin sits at the joint pivot, not the box center)
    joint0_pos = np.array([0.0, 0.0,  0.0])
    joint1_pos = np.array([0.0, 0.30, 0.0])

    fig = go.Figure()

    # boxes — different colors to distinguish
    add_box(fig, center=link0_origin, half_extents=half_extents, color='steelblue',  name='link_0')
    add_box(fig, center=link1_origin, half_extents=half_extents, color='darkorange', name='link_1')

    # link frames at box centers (old convention)
    add_frame(fig, link0_origin, name='link_0 (box center)')
    add_frame(fig, link1_origin, name='link_1 (box center)')

    # link frames at joint pivots (slides convention) — larger arrows to distinguish
    add_frame(fig, joint0_pos, axis_len=0.08, name='link_0 (joint pivot)')
    add_frame(fig, joint1_pos, axis_len=0.08, name='link_1 (joint pivot)')

    # world frame (dashed)
    for d, color in [([1,0,0],'red'), ([0,1,0],'green'), ([0,0,1],'blue')]:
        tip = np.array(d, dtype=float) * 0.03
        fig.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode='lines', line=dict(color=color, width=3, dash='dash'),
            showlegend=False,
        ))

    # joint pivot markers
    add_joint(fig, joint0_pos, name='joint_0')
    add_joint(fig, joint1_pos, name='joint_1')

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        title='small frames = box centers, large frames = joint pivots (slides convention)',
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()