"""
visualize.py
------------
Visualizes raw GPS vs Kalman filtered output for 11 players.

  1. Snapshot      — raw GPS cloud vs filtered positions on pitch
  2. RMSE by role  — filter improvement per player
  3. Centroid      — raw vs filtered team centroid over 45 min
  4. Heatmaps      — raw vs filtered per player

Usage
-----
    python visualize.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

from simulate import simulate, add_gps_noise, PLAYERS, N_STEPS, DT
from filter   import run_all, filtered_centroid, raw_centroid

os.makedirs('outputs', exist_ok=True)

WHITE = '#FFFFFF'
BG    = '#111827'
SNAP  = N_STEPS // 2

ROLE_COLORS = {
    'GK': '#FFD700', 'CB': '#4daf4a', 'LB': '#4daf4a', 'RB': '#4daf4a',
    'CM': '#377eb8', 'LW': '#ff7f00', 'RW': '#ff7f00', 'ST': '#e41a1c',
}


def draw_pitch(ax, alpha=1.0):
    ax.set_facecolor('#2d6a2d')
    PL, PW, lw = 105, 68, 1.5
    def box(x0, y0, w, h):
        ax.plot([x0,x0+w,x0+w,x0,x0],[y0,y0,y0+h,y0+h,y0],
                color=WHITE, lw=lw, alpha=alpha)
    box(0, 0, PL, PW)
    ax.plot([PL/2]*2, [0, PW], color=WHITE, lw=lw, alpha=alpha)
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(PL/2+9.15*np.cos(th), PW/2+9.15*np.sin(th),
            color=WHITE, lw=lw, alpha=alpha)
    for x0, sgn in [(0,1),(PL,-1)]:
        box(x0, PW/2-20.16, sgn*16.5, 40.32)
        box(x0, PW/2- 9.16, sgn* 5.5, 18.32)
        ax.plot([x0,x0+sgn*(-2),x0+sgn*(-2),x0],
                [PW/2-3.66,PW/2-3.66,PW/2+3.66,PW/2+3.66],
                color=WHITE, lw=2.5, alpha=alpha)
    ax.set_xlim(-5, PL+5); ax.set_ylim(-5, PW+5)
    ax.set_aspect('equal'); ax.axis('off')


def cov_ellipse(ax, mean, cov, n_std=2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    idx  = vals.argsort()[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    ang  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=ang, **kw))


# ── Figure 1: Snapshot — raw vs filtered ──────────────────────────────────────

def plot_snapshot(positions, noisy, filtered):
    fig, (ax_raw, ax_filt) = plt.subplots(1, 2, figsize=(16, 7),
                                           facecolor=BG)
    fig.suptitle(
        f'L0 Raw GPS vs L1 Kalman Filter  |  t = {SNAP*DT/60:.1f} min  |  4-3-3',
        color=WHITE, fontsize=11, y=1.00)

    for ax, title in [(ax_raw, 'L0 — Raw GPS'), (ax_filt, 'L1 — Kalman Filter')]:
        draw_pitch(ax)
        ax.set_title(title, color=WHITE, fontsize=10)

    # Formation lines helper
    def formation_lines(ax, pts):
        for line in [['LB','CB1','CB2','RB'],['LCM','CM','RCM'],['LW','ST','RW']]:
            ax.plot(*np.array([pts[n] for n in line]).T,
                    color=WHITE, lw=0.8, alpha=0.3, zorder=2)

    snap_true  = {p[0]: positions[p[0]][SNAP] for p in PLAYERS}
    snap_noisy = {p[0]: noisy[p[0]][SNAP]     for p in PLAYERS}
    snap_filt  = {p[0]: filtered[p[0]]['xs'][SNAP, :2] for p in PLAYERS}
    snap_P     = {p[0]: filtered[p[0]]['Ps'][SNAP, :2, :2] for p in PLAYERS}

    for name, role, *_ in PLAYERS:
        col = ROLE_COLORS[role]

        # Raw panel
        gps_cloud = noisy[name][max(0, SNAP-60):SNAP]
        ax_raw.scatter(*gps_cloud.T, s=8, c=col, alpha=0.20, zorder=3)
        ax_raw.scatter(*snap_noisy[name], s=90, c=col,
                       edgecolors=WHITE, linewidths=0.8, zorder=5)

        # Filtered panel
        ax_filt.scatter(*gps_cloud.T, s=8, c=col, alpha=0.10, zorder=2)
        kf_trail = filtered[name]['xs'][max(0,SNAP-100):SNAP, :2]
        ax_filt.plot(*kf_trail.T, color=col, lw=1.2, alpha=0.6, zorder=3)
        ax_filt.scatter(*snap_filt[name], s=90, c=col,
                        edgecolors=WHITE, linewidths=0.8, zorder=5)
        cov_ellipse(ax_filt, snap_filt[name], snap_P[name],
                    alpha=0.18, color=col, zorder=4)
        ax_filt.text(snap_filt[name][0], snap_filt[name][1]+2.8,
                     name, ha='center', color=col,
                     fontsize=7, fontweight='bold', zorder=6)

    formation_lines(ax_raw,  snap_noisy)
    formation_lines(ax_filt, snap_filt)

    legend_els = [
        Line2D([0],[0], marker='o', color='w', label='GK',
               markerfacecolor=ROLE_COLORS['GK'], markersize=8),
        Line2D([0],[0], marker='o', color='w', label='Defenders',
               markerfacecolor=ROLE_COLORS['CB'], markersize=8),
        Line2D([0],[0], marker='o', color='w', label='Midfielders',
               markerfacecolor=ROLE_COLORS['CM'], markersize=8),
        Line2D([0],[0], marker='o', color='w', label='Attackers',
               markerfacecolor=ROLE_COLORS['ST'], markersize=8),
    ]
    ax_filt.legend(handles=legend_els, fontsize=8, facecolor=BG,
                   edgecolor=WHITE, labelcolor=WHITE, loc='upper left')
    ax_filt.text(52.5, -3.5, 'Ellipses = 2σ covariance',
                 ha='center', color='#aaa', fontsize=7.5)

    fig.tight_layout()
    fig.savefig('outputs/01_snapshot.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 01_snapshot.png')


# ── Figure 2: RMSE by player ──────────────────────────────────────────────────

def plot_rmse(positions, noisy, filtered):
    names, raw_rmse, kf_rmse, colors = [], [], [], []

    for name, role, *_ in PLAYERS:
        true = positions[name]
        raw  = noisy[name]
        filt = filtered[name]['xs'][:, :2]
        names.append(name)
        raw_rmse.append(float(np.sqrt(((raw  - true)**2).mean())))
        kf_rmse.append( float(np.sqrt(((filt - true)**2).mean())))
        colors.append(ROLE_COLORS[role])

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor('#1a2035')
    ax.bar(x - w/2, raw_rmse, w, label='L0 Raw GPS',
           color='#888', edgecolor=WHITE, linewidth=0.5)
    ax.bar(x + w/2, kf_rmse,  w, label='L1 Kalman filter',
           color=colors, edgecolor=WHITE, linewidth=0.5)

    for i, (r, k) in enumerate(zip(raw_rmse, kf_rmse)):
        impr = 100 * (1 - k/r)
        col  = '#4daf4a' if impr >= 0 else '#e41a1c'
        ax.text(i, max(r, k) + 0.02, f'{impr:+.0f}%',
                ha='center', color=col, fontsize=7.5, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(names, color=WHITE, fontsize=9)
    ax.set_ylabel('Position RMSE [m]', color=WHITE)
    ax.set_title('Filter Improvement by Player  |  4-3-3',
                 color=WHITE, fontsize=10)
    ax.tick_params(axis='y', colors=WHITE)
    for sp in ax.spines.values(): sp.set_color('#444')
    ax.legend(fontsize=9, facecolor=BG, edgecolor=WHITE, labelcolor=WHITE)

    fig.tight_layout()
    fig.savefig('outputs/02_rmse.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 02_rmse.png')


# ── Figure 3: Centroid — raw vs filtered ──────────────────────────────────────

def plot_centroid(positions, noisy, filtered, centroid):
    cent_raw  = raw_centroid(noisy)
    cent_filt = filtered_centroid(filtered)
    true_cent = np.mean([positions[p[0]] for p in PLAYERS], axis=0)
    t = np.arange(N_STEPS) * DT / 60.0

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(13, 6),
                                      sharex=True, facecolor=BG)
    fig.suptitle('Team Centroid  |  Raw GPS vs Kalman Filter  |  4-3-3',
                 color=WHITE, fontsize=10)

    for ax, dim, label in [(ax_x, 0, 'x [m]'), (ax_y, 1, 'y [m]')]:
        ax.set_facecolor('#1a2035')
        ax.plot(t, cent_raw[:, dim],  color='#888', lw=0.8,
                alpha=0.7, label='L0 raw centroid')
        ax.plot(t, true_cent[:, dim], color=WHITE, lw=1.2,
                alpha=0.5, ls='--', label='True centroid')
        ax.plot(t, cent_filt[:, dim], color='#4daf4a', lw=1.8,
                label='L1 filtered centroid')
        ax.set_ylabel(label, color=WHITE)
        ax.tick_params(colors=WHITE)
        ax.grid(alpha=0.15, color=WHITE)
        for sp in ax.spines.values(): sp.set_color('#444')

    ax_x.legend(fontsize=8.5, facecolor=BG, edgecolor=WHITE,
                labelcolor=WHITE, loc='upper right')
    ax_y.set_xlabel('Match time [min]', color=WHITE)

    fig.tight_layout()
    fig.savefig('outputs/03_centroid.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 03_centroid.png')


# ── Figure 4: Heatmaps — raw vs filtered ─────────────────────────────────────

def plot_heatmaps(positions, noisy, filtered):
    fig, axes = plt.subplots(2, 11, figsize=(22, 6), facecolor=BG)

    for col_idx, (name, role, *_) in enumerate(PLAYERS):
        col  = ROLE_COLORS[role]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            '', ['#1a3a1a', col])

        for row_idx, (data, title) in enumerate([
                (noisy[name],                    'Raw'),
                (filtered[name]['xs'][:, :2],    'KF'),
        ]):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor('#1a3a1a')
            H, _, _ = np.histogram2d(data[:,0], data[:,1],
                                     bins=[52,34], range=[[0,105],[0,68]])
            H = gaussian_filter(H.T, sigma=2.0)
            ax.imshow(H, extent=[0,105,0,68], origin='lower',
                      aspect='equal', cmap=cmap, alpha=0.9)
            ax.plot([0,105,105,0,0],[0,0,68,68,0],
                    color=WHITE, lw=0.7, alpha=0.4)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f'{name}\n({role})', color=col,
                             fontsize=6.5, fontweight='bold', pad=2)
            if col_idx == 0:
                ax.text(-12, 34, title, color=WHITE, fontsize=8,
                        fontweight='bold', va='center', rotation=90)

    fig.suptitle('Player Heatmaps  |  Raw GPS (top) vs Kalman Filter (bottom)  |  4-3-3',
                 color=WHITE, fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig('outputs/04_heatmaps.png', dpi=130,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print('Saved 04_heatmaps.png')


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    print('Simulating...')
    positions, centroid, t = simulate()
    noisy    = add_gps_noise(positions)

    print('Filtering...')
    filtered = run_all(noisy)

    print('Generating figures...')
    plot_snapshot(positions, noisy, filtered)
    plot_rmse(positions, noisy, filtered)
    plot_centroid(positions, noisy, filtered, centroid)
    plot_heatmaps(positions, noisy, filtered)
