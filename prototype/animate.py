"""
animate.py
----------
Animates Voronoi territories using Kalman-filtered player positions.

Comparing L0 (raw GPS) vs L1 (filtered) shows how the filter stabilises
territorial control — regions shift less erratically, and the team
centroid trail is smoother.

1 frame per second, each frame = 4.5 seconds of match time.

Usage
-----
    python animate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi

from simulate import simulate, add_gps_noise, PLAYERS, N_STEPS, DT
from filter   import run_all, filtered_centroid, raw_centroid

os.makedirs('outputs', exist_ok=True)

WHITE  = '#FFFFFF'
BG     = '#111827'

ROLE_COLORS = {
    'GK': '#FFD700', 'CB': '#4daf4a', 'LB': '#4daf4a', 'RB': '#4daf4a',
    'CM': '#377eb8', 'LW': '#ff7f00', 'RW': '#ff7f00', 'ST': '#e41a1c',
}

STRIDE = 45
FPS    = 1


def draw_pitch(ax):
    ax.set_facecolor('#1a3a1a')
    PL, PW, lw = 105, 68, 1.0
    def box(x0, y0, w, h):
        ax.plot([x0,x0+w,x0+w,x0,x0],[y0,y0,y0+h,y0+h,y0],
                color=WHITE, lw=lw, alpha=0.4)
    box(0, 0, PL, PW)
    ax.plot([PL/2]*2, [0, PW], color=WHITE, lw=lw, alpha=0.4)
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(PL/2+9.15*np.cos(th), PW/2+9.15*np.sin(th),
            color=WHITE, lw=lw, alpha=0.4)
    for x0, sgn in [(0,1),(PL,-1)]:
        box(x0, PW/2-20.16, sgn*16.5, 40.32)
        ax.plot([x0,x0+sgn*(-2),x0+sgn*(-2),x0],
                [PW/2-3.66,PW/2-3.66,PW/2+3.66,PW/2+3.66],
                color=WHITE, lw=1.5, alpha=0.4)
    ax.set_xlim(-3, PL+3); ax.set_ylim(-3, PW+3)
    ax.set_aspect('equal'); ax.axis('off')


def voronoi_ridges(pts):
    vor    = Voronoi(pts)
    center = pts.mean(axis=0)
    segs   = []
    for (p1, p2), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in ridge_verts:
            v1 = vor.vertices[ridge_verts[0]]
            v2 = vor.vertices[ridge_verts[1]]
        else:
            finite = ridge_verts[1] if ridge_verts[0]==-1 else ridge_verts[0]
            v1     = vor.vertices[finite]
            tang   = pts[p2] - pts[p1]
            norm   = np.array([-tang[1], tang[0]])
            norm  /= np.linalg.norm(norm)
            if np.dot(norm, (pts[p1]+pts[p2])/2 - center) < 0:
                norm = -norm
            v2 = v1 + norm * 150
        if (max(v1[0],v2[0])>-5 and min(v1[0],v2[0])<110 and
                max(v1[1],v2[1])>-5 and min(v1[1],v2[1])<73):
            segs.append([v1, v2])
    return segs


def make_animation(positions, noisy, filtered):
    frames       = list(range(0, N_STEPS, STRIDE))
    n_frames     = len(frames)
    player_names = [p[0] for p in PLAYERS]
    colors       = [ROLE_COLORS[p[1]] for p in PLAYERS]

    # Position arrays: (11, N, 2)
    raw_pos  = np.array([noisy[n]                      for n in player_names])
    filt_pos = np.array([filtered[n]['xs'][:, :2]      for n in player_names])
    cent_raw  = raw_centroid(noisy)
    cent_filt = filtered_centroid(filtered)

    print(f'Precomputing {n_frames} frames...')
    raw_ridges  = [voronoi_ridges(raw_pos[:,  f, :]) for f in frames]
    filt_ridges = [voronoi_ridges(filt_pos[:, f, :]) for f in frames]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 7.5), facecolor=BG)
    fig.suptitle(
        '4-3-3 Kalman Filter Tracking  |  Norway vs France  |  FIFA World Cup 2026',
        color=WHITE, fontsize=11, y=1.00)

    for ax, title in [(ax_l, 'L0 — Raw GPS'), (ax_r, 'L1 — Kalman Filter')]:
        draw_pitch(ax)
        ax.set_title(title, color=WHITE, fontsize=10, pad=6)

    # Voronoi ridges
    raw_coll  = LineCollection(raw_ridges[0],  colors=WHITE, lw=0.8, alpha=0.45, zorder=2)
    filt_coll = LineCollection(filt_ridges[0], colors=WHITE, lw=0.8, alpha=0.45, zorder=2)
    ax_l.add_collection(raw_coll)
    ax_r.add_collection(filt_coll)

    # Player dots
    raw_sc   = [ax_l.scatter([], [], s=75, c=col, edgecolors=WHITE,
                              linewidths=0.7, zorder=4) for col in colors]
    filt_sc  = [ax_r.scatter([], [], s=75, c=col, edgecolors=WHITE,
                              linewidths=0.7, zorder=4) for col in colors]

    # Centroid
    cr_sc = ax_l.scatter([], [], s=180, marker='D', c='#FFD700',
                          edgecolors=WHITE, linewidths=1.5, zorder=6)
    cf_sc = ax_r.scatter([], [], s=180, marker='D', c='#FFD700',
                          edgecolors=WHITE, linewidths=1.5, zorder=6)
    cr_trail, = ax_l.plot([], [], color='#FFD700', lw=1.5, alpha=0.4, zorder=3)
    cf_trail, = ax_r.plot([], [], color='#FFD700', lw=1.5, alpha=0.4, zorder=3)

    # Formation lines
    LINES = [['LB','CB1','CB2','RB'], ['LCM','CM','RCM'], ['LW','ST','RW']]
    idx   = {p[0]: i for i, p in enumerate(PLAYERS)}
    raw_lines  = [ax_l.plot([], [], color=WHITE, lw=0.8, alpha=0.3)[0] for _ in LINES]
    filt_lines = [ax_r.plot([], [], color=WHITE, lw=0.8, alpha=0.3)[0] for _ in LINES]

    time_text = fig.text(0.5, 0.01, '', ha='center', color=WHITE,
                          fontsize=11, fontweight='bold')

    legend_els = [Line2D([0],[0], marker='o', color='w', label=r,
                          markerfacecolor=ROLE_COLORS[r], markersize=8)
                  for r in ['GK','CB','CM','ST']]
    legend_els += [Line2D([0],[0], marker='D', color='w', label='Centroid',
                           markerfacecolor='#FFD700', markersize=8)]
    ax_r.legend(handles=legend_els, fontsize=8, facecolor=BG,
                edgecolor=WHITE, labelcolor=WHITE, loc='upper left')

    def update(fi):
        step = frames[fi]

        raw_coll.set_segments(raw_ridges[fi])
        filt_coll.set_segments(filt_ridges[fi])

        for i, (rs, fs) in enumerate(zip(raw_sc, filt_sc)):
            rs.set_offsets([raw_pos[i, step]])
            fs.set_offsets([filt_pos[i, step]])

        cr_sc.set_offsets([cent_raw[step]])
        cf_sc.set_offsets([cent_filt[step]])

        trail_start = max(0, step - 600)
        cr_trail.set_data(*cent_raw[trail_start:step+1].T)
        cf_trail.set_data(*cent_filt[trail_start:step+1].T)

        for line_names, rl, fl in zip(LINES, raw_lines, filt_lines):
            rl.set_data(*np.array([raw_pos[idx[n], step] for n in line_names]).T)
            fl.set_data(*np.array([filt_pos[idx[n], step] for n in line_names]).T)

        time_text.set_text(f"{step * DT / 60:.1f}'")

        return ([raw_coll, filt_coll, cr_sc, cf_sc,
                 cr_trail, cf_trail, time_text]
                + raw_sc + filt_sc + raw_lines + filt_lines)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000/FPS, blit=True)

    out = 'outputs/filter_animation.mp4'
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2500)
    print(f'Saving {n_frames} frames at {FPS} fps...')
    ani.save(out, writer=writer, dpi=110,
             savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    print('Simulating...')
    positions, centroid, t = simulate()
    noisy    = add_gps_noise(positions)

    print('Filtering...')
    filtered = run_all(noisy)

    make_animation(positions, noisy, filtered)
