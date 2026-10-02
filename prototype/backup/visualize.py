"""
visualize.py
------------
Four visualizations of the 45-minute possession simulation.

  1. Pitch snapshot     — players, Voronoi control, passing network
  2. Possession timeline — which team held the ball over 45 min
  3. Pass map           — where passes were attempted and their outcome
  4. Player heatmaps    — where each player spent time on the pitch

Usage
-----
    python visualize.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from scipy.ndimage import gaussian_filter

from simulate import (simulate, add_gps_noise, voronoi_areas,
                      PLAYERS, PLAYER_TEAM, _GRID, _gx, _gy,
                      N_STEPS, DT)

os.makedirs('outputs', exist_ok=True)

NOR_RED  = '#EF2B2D'
FRA_BLUE = '#003189'
WHITE    = '#FFFFFF'
BG       = '#111827'
SNAP     = N_STEPS // 2   # snapshot at halftime


# ── Pitch ────────────────────────────────────────────────────────────────────

def draw_pitch(ax, alpha=1.0):
    ax.set_facecolor('#2d6a2d')
    PL, PW, lw = 105, 68, 1.5

    def box(x0, y0, w, h):
        ax.plot([x0, x0+w, x0+w, x0, x0],
                [y0, y0, y0+h, y0+h, y0],
                color=WHITE, lw=lw, alpha=alpha)

    box(0, 0, PL, PW)
    ax.plot([PL/2]*2, [0, PW], color=WHITE, lw=lw, alpha=alpha)
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(PL/2 + 9.15*np.cos(th), PW/2 + 9.15*np.sin(th),
            color=WHITE, lw=lw, alpha=alpha)
    ax.plot(PL/2, PW/2, 'o', color=WHITE, ms=3, alpha=alpha)

    for x0, sgn in [(0, 1), (PL, -1)]:
        box(x0, PW/2-20.16, sgn*16.5, 40.32)
        box(x0, PW/2- 9.16, sgn* 5.5, 18.32)
        ax.plot(x0+sgn*11, PW/2, 'o', color=WHITE, ms=3, alpha=alpha)
        ax.plot([x0, x0+sgn*(-2), x0+sgn*(-2), x0],
                [PW/2-3.66, PW/2-3.66, PW/2+3.66, PW/2+3.66],
                color=WHITE, lw=2.5, alpha=alpha)

    ax.set_xlim(-5, PL+5)
    ax.set_ylim(-5, PW+5)
    ax.set_aspect('equal')
    ax.axis('off')


# ── Figure 1: Pitch snapshot ──────────────────────────────────────────────────

def plot_snapshot(positions, noisy, possession, passes):
    snap_pos  = {n: positions[n][SNAP] for n in positions}
    possessor = possession[SNAP]

    fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
    ax  = fig.add_subplot(111)
    draw_pitch(ax)

    # Voronoi territorial control
    all_pts = np.array([snap_pos[p[0]] for p in PLAYERS])
    dists   = np.linalg.norm(_GRID[:, None] - all_pts[None], axis=2)
    nearest = dists.argmin(axis=1)
    teams   = [PLAYER_TEAM[p[0]] for p in PLAYERS]
    ctrl    = np.array([teams[n] for n in nearest])

    for team, col in [('NOR', NOR_RED), ('FRA', FRA_BLUE)]:
        mask = (ctrl == team).reshape(_gx.shape)
        ax.contourf(_gx, _gy, mask.astype(float),
                    levels=[0.5, 1.5], colors=[col], alpha=0.18, zorder=1)

    # Passing network (successful passes up to snap)
    pass_counts = {}
    for p in passes:
        if p['step'] <= SNAP and p['success']:
            edge = (p['frm'], p['to'])
            pass_counts[edge] = pass_counts.get(edge, 0) + 1

    max_count = max(pass_counts.values(), default=1)
    for (frm, to), count in pass_counts.items():
        a   = snap_pos[frm]
        b   = snap_pos[to]
        col = NOR_RED if PLAYER_TEAM[frm] == 'NOR' else FRA_BLUE
        ax.annotate('', xy=b, xytext=a,
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=0.8 + 2.0*count/max_count,
                                    alpha=0.5, mutation_scale=10),
                    zorder=3)

    # Players
    for name, team, *_ in PLAYERS:
        col = NOR_RED if team == 'NOR' else FRA_BLUE
        p   = snap_pos[name]
        gps = noisy[name][max(0, SNAP-30):SNAP]

        ax.scatter(*gps.T, s=7, c=col, alpha=0.20, zorder=3)
        ax.plot(*positions[name][:SNAP].T,
                color=col, lw=0.6, alpha=0.20, zorder=3)

        size = 130 if name == possessor else 85
        lw   = 2.5 if name == possessor else 0.8
        ax.scatter(*p, s=size, c=col,
                   edgecolors=WHITE, linewidths=lw, zorder=5)
        ax.text(p[0], p[1]+2.8, name[:7],
                ha='center', color=col,
                fontsize=6.5, fontweight='bold', zorder=6)

    legend_els = [
        Line2D([0],[0], marker='o', color='w', label='Norway',
               markerfacecolor=NOR_RED,  markersize=9),
        Line2D([0],[0], marker='o', color='w', label='France',
               markerfacecolor=FRA_BLUE, markersize=9),
        Line2D([0],[0], marker='o', color='w',
               label=f'Ball  ({possessor})',
               markerfacecolor=WHITE, markersize=11),
        Line2D([0],[0], color='grey', lw=1.5, alpha=0.5,
               label='Passing network'),
    ]
    ax.legend(handles=legend_els, loc='upper left', fontsize=9,
              facecolor=BG, edgecolor=WHITE, labelcolor=WHITE,
              framealpha=0.9)
    ax.set_title(
        f'Pitch Snapshot  t = {SNAP*DT/60:.1f} min  |  '
        'Norway vs France  |  FIFA World Cup 2026',
        color=WHITE, fontsize=10, pad=8)
    ax.text(52.5, -3.5,
            'Shading = Voronoi control  |  '
            'Arrows = passing network  |  '
            'Large circle = ball carrier',
            ha='center', color='#aaa', fontsize=7.5)

    fig.tight_layout()
    fig.savefig('outputs/01_snapshot.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 01_snapshot.png')


# ── Figure 2: Possession timeline ─────────────────────────────────────────────

def plot_possession(possession, passes):
    t    = np.arange(N_STEPS) * DT / 60.0
    ctrl = np.array([1 if PLAYER_TEAM[p] == 'NOR' else -1
                     for p in possession], dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
    ax.set_facecolor('#1a2035')

    ax.fill_between(t, ctrl, where=ctrl > 0,
                    color=NOR_RED,  alpha=0.65, label='Norway')
    ax.fill_between(t, ctrl, where=ctrl < 0,
                    color=FRA_BLUE, alpha=0.65, label='France')
    ax.axhline(0, color=WHITE, lw=0.8, alpha=0.3)

    for p in passes:
        col = WHITE if p['success'] else '#ff6b6b'
        ls  = '-' if p['success'] else '--'
        ax.axvline(p['step'] * DT / 60.0,
                   color=col, lw=0.6, alpha=0.35, ls=ls)

    n_succ = sum(1 for p in passes if p['success'])
    nor_t  = (np.array(ctrl) == 1).mean() * 100
    fra_t  = (np.array(ctrl) == -1).mean() * 100

    ax.set_xlim(0, N_STEPS * DT / 60.0)
    ax.set_ylim(-1.4, 1.6)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['France', 'Norway'], color=WHITE, fontsize=9)
    ax.set_xlabel('Match time [min]', color=WHITE)
    ax.tick_params(axis='x', colors=WHITE)
    for sp in ax.spines.values(): sp.set_color('#444')

    ax.set_title(
        f'Possession Timeline  |  '
        f'NOR {nor_t:.0f}%  FRA {fra_t:.0f}%  |  '
        f'{len(passes)} passes  |  {n_succ} successful ({100*n_succ//len(passes)}%)  |  '
        f'white = success  red = turnover',
        color=WHITE, fontsize=9)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=WHITE,
              labelcolor=WHITE, loc='upper right')

    fig.tight_layout()
    fig.savefig('outputs/02_possession.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 02_possession.png')


# ── Figure 3: Pass map ────────────────────────────────────────────────────────

def plot_pass_map(positions, passes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)

    for ax, team, col, label in [
            (axes[0], 'NOR', NOR_RED,  'Norway'),
            (axes[1], 'FRA', FRA_BLUE, 'France')]:
        draw_pitch(ax, alpha=0.7)
        ax.set_title(f'{label} Pass Map', color=WHITE, fontsize=10)

        for p in passes:
            if PLAYER_TEAM[p['frm']] != team:
                continue
            step = p['step']
            a    = positions[p['frm']][step]
            b    = positions[p['to']][step]
            c    = col if p['success'] else '#ff6b6b'
            al   = 0.5 if p['success'] else 0.7
            lw   = 1.0 if p['success'] else 1.2
            ax.annotate('', xy=b, xytext=a,
                        arrowprops=dict(arrowstyle='->',
                                        color=c, lw=lw,
                                        alpha=al, mutation_scale=8),
                        zorder=4)
            ax.scatter(*a, s=20, c=c, alpha=al, zorder=5)

        n_team    = sum(1 for p in passes if PLAYER_TEAM[p['frm']] == team)
        n_success = sum(1 for p in passes
                        if PLAYER_TEAM[p['frm']] == team and p['success'])
        ax.text(52.5, -3.5,
                f'{n_success}/{n_team} completed  '
                f'({100*n_success//max(n_team,1)}%)',
                ha='center', color='#aaa', fontsize=8)

    legend_els = [
        Line2D([0],[0], color=WHITE,    lw=1.5, label='Successful pass'),
        Line2D([0],[0], color='#ff6b6b', lw=1.5, label='Turnover'),
    ]
    axes[0].legend(handles=legend_els, fontsize=8.5,
                   facecolor=BG, edgecolor=WHITE,
                   labelcolor=WHITE, loc='upper left')

    fig.suptitle('Pass Map  |  Norway vs France  |  FIFA World Cup 2026',
                 color=WHITE, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig('outputs/03_pass_map.png', dpi=130,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Saved 03_pass_map.png')


# ── Figure 4: Player heatmaps ─────────────────────────────────────────────────

def plot_heatmaps(positions):
    nor_players = [p[0] for p in PLAYERS if p[1] == 'NOR']
    fra_players = [p[0] for p in PLAYERS if p[1] == 'FRA']

    fig, axes = plt.subplots(2, 11, figsize=(22, 7), facecolor=BG)

    for row, (players, team, cmap) in enumerate([
            (nor_players, 'Norway', 'Reds'),
            (fra_players, 'France', 'Blues')]):
        for col, name in enumerate(players):
            ax = axes[row][col]
            ax.set_facecolor('#1a3a1a')

            pos = positions[name]
            H, xe, ye = np.histogram2d(
                pos[:, 0], pos[:, 1],
                bins=[52, 34],
                range=[[0, 105], [0, 68]])
            H = gaussian_filter(H.T, sigma=2.0)
            ax.imshow(H, extent=[0, 105, 0, 68], origin='lower',
                      aspect='equal', cmap=cmap, alpha=0.85)

            # Pitch outline only
            ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0],
                    color=WHITE, lw=0.8, alpha=0.5)
            ax.plot([52.5]*2, [0, 68], color=WHITE, lw=0.6, alpha=0.3)
            ax.set_xlim(-2, 107); ax.set_ylim(-2, 70)
            ax.axis('off')
            ax.set_title(name[:8], color=WHITE, fontsize=7,
                         fontweight='bold', pad=2)

    axes[0][0].text(-15, 34, 'Norway', color=NOR_RED, fontsize=10,
                    fontweight='bold', va='center', rotation=90)
    axes[1][0].text(-15, 34, 'France', color=FRA_BLUE, fontsize=10,
                    fontweight='bold', va='center', rotation=90)

    fig.suptitle(
        'Player Heatmaps  |  Norway vs France  |  FIFA World Cup 2026',
        color=WHITE, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig('outputs/04_heatmaps.png', dpi=130,
                facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print('Saved 04_heatmaps.png')


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Simulating...')
    positions, possession, passes = simulate(seed=None)
    noisy = add_gps_noise(positions)

    print('Generating figures...')
    plot_snapshot(positions, noisy, possession, passes)
    plot_possession(possession, passes)
    plot_pass_map(positions, passes)
    plot_heatmaps(positions)
