#!/usr/bin/env python3
"""
Norway vs France — FIFA World Cup 2026, Group I (June 26, 2026)
22-Player Match Tracking — Minimal Simulation
==============================================
One independent Kalman filter per player, applied to simulated GPS data
from a 15-second attacking phase.  Each player's velocity follows an
Ornstein-Uhlenbeck process, consistent with the DWNA process noise model
used in the Kalman filter.

Processing Levels
-----------------
  L0  Raw GPS measurements     (noisy positions, σ = 0.8 m)
  L1  Kalman filter estimates  (position + latent velocity, per player)
  L2  Speed distribution       (derived from L1 velocity state)
"""

import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np


# ============================================================================
# Player definitions
# (name, team, x0, y0, vx_drift, vy_drift, sigma_a)
# Norway attacks right (toward x=105). Right touchline = low y.
# ============================================================================

PLAYERS = [
    # ── Norway ──────────────────────────────────────────────────────────
    ('Nyland',      'NOR',  8, 34,  0.0,  0.0, 0.5),   # GK
    ('Ryerson',     'NOR', 42, 58,  2.2,  0.8, 2.0),   # RB
    ('H-Olsen',     'NOR', 38, 42,  1.0,  0.0, 1.5),   # CB
    ('Ostigard',    'NOR', 38, 26,  1.0,  0.0, 1.5),   # CB
    ('Meling',      'NOR', 44, 10,  1.8, -0.5, 2.0),   # LB
    ('Aursnes',     'NOR', 60, 18,  2.2,  0.3, 2.5),   # LCM
    ('Berge',       'NOR', 58, 34,  2.5,  0.0, 2.5),   # CM
    ('Odegaard',    'NOR', 65, 50,  2.8,  0.5, 2.5),   # RCM
    ('Hauge',       'NOR', 72, 12,  2.0, -1.5, 3.0),   # LW
    ('Haaland',     'NOR', 55, 34,  6.0,  1.0, 3.5),   # ST — sprint
    ('Nusa',        'NOR', 74, 58,  2.5, -3.0, 3.0),   # RW — cut inside
    # ── France ──────────────────────────────────────────────────────────
    ('Maignan',     'FRA',100, 34,  0.0,  0.0, 0.5),   # GK
    ('Kounde',      'FRA', 84, 58, -0.8,  0.3, 2.0),   # RB
    ('Konate',      'FRA', 83, 42, -0.5,  2.0, 2.0),   # CB — tracks Haaland
    ('Saliba',      'FRA', 83, 26, -0.5, -0.5, 2.0),   # CB
    ('T-Hernandez', 'FRA', 84, 10, -0.8, -0.5, 2.0),   # LB
    ('Griezmann',   'FRA', 73, 48, -1.5,  0.3, 2.5),   # AM
    ('Tchouameni',  'FRA', 72, 34, -1.2,  0.0, 2.5),   # CM
    ('Kante',       'FRA', 72, 22, -1.2,  0.0, 2.5),   # CM
    ('Dembele',     'FRA', 76, 14, -1.0, -1.5, 3.0),   # LW
    ('Mbappe',      'FRA', 80, 34, -2.8,  0.8, 3.0),   # ST — high press
    ('Olise',       'FRA', 80, 54, -1.2, -2.0, 3.0),   # RW
]

NOR_RED  = '#EF2B2D'
FRA_BLUE = '#003189'
GOLD     = '#FFD700'
WHITE    = '#FFFFFF'
BG       = '#111827'
TEAM_COLORS = {'NOR': NOR_RED, 'FRA': FRA_BLUE}


# ============================================================================
# Pitch
# ============================================================================

def draw_pitch(ax, pitch_color='#2d6a2d', line_color='white'):
    ax.set_facecolor(pitch_color)
    PL, PW, lw = 105, 68, 1.5

    def box(x0, y0, w, h):
        ax.plot([x0, x0+w, x0+w, x0, x0],
                [y0, y0, y0+h, y0+h, y0], color=line_color, lw=lw)

    box(0, 0, PL, PW)
    ax.plot([PL/2]*2, [0, PW], color=line_color, lw=lw)
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(PL/2 + 9.15*np.cos(th), PW/2 + 9.15*np.sin(th),
            color=line_color, lw=lw)
    ax.plot(PL/2, PW/2, 'o', color=line_color, ms=3)

    for x0, sgn in [(0, 1), (PL, -1)]:
        box(x0, PW/2-20.16, sgn*16.5, 40.32)
        box(x0, PW/2- 9.16, sgn* 5.5, 18.32)
        ax.plot(x0+sgn*11, PW/2, 'o', color=line_color, ms=3)
        ax.plot([x0, x0+sgn*(-2), x0+sgn*(-2), x0],
                [PW/2-3.66]*2 + [PW/2+3.66]*2,
                color=line_color, lw=2.5)

    ax.set_xlim(-5, PL+5); ax.set_ylim(-5, PW+5)
    ax.set_aspect('equal'); ax.axis('off')


# ============================================================================
# Kalman filter
# ============================================================================

class KalmanFilter:
    """
    2-D constant-velocity Kalman filter.

    State   x = [px, py, vx, vy]^T
    Predict x_{k|k-1} = F x_{k-1}
            P_{k|k-1} = F P_{k-1} F^T + Q
    Update  K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
            x_{k|k} = x_{k|k-1} + K (z - H x_{k|k-1})
            P_{k|k} = (I - KH) P_{k|k-1}

    Assumptions
    -----------
    - Constant-velocity (CV) dynamics
    - Position-only observations; velocity is a latent state
    - DWNA process noise model (Bar-Shalom et al., 2001)
    - Zero-mean white Gaussian process and measurement noise
    """

    def __init__(self, dt, sigma_q, sigma_r):
        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        q = sigma_q ** 2
        self.Q = q * np.array([
            [dt**4/4,       0, dt**3/2,       0],
            [      0, dt**4/4,       0, dt**3/2],
            [dt**3/2,       0,   dt**2,       0],
            [      0, dt**3/2,       0,   dt**2]])
        self.R = (sigma_r ** 2) * np.eye(2)

    def run(self, measurements):
        N  = len(measurements)
        xs = np.zeros((N, 4))
        Ps = np.zeros((N, 4, 4))
        x  = np.array([measurements[0, 0], measurements[0, 1], 0., 0.])
        P  = np.diag([1., 1., 25., 25.])
        for k, z in enumerate(measurements):
            x_p = self.F @ x
            P_p = self.F @ P @ self.F.T + self.Q
            y   = z - self.H @ x_p
            S   = self.H @ P_p @ self.H.T + self.R
            K   = P_p @ self.H.T @ np.linalg.inv(S)
            x   = x_p + K @ y
            P   = (np.eye(4) - K @ self.H) @ P_p
            xs[k], Ps[k] = x, P
        return xs, Ps


# ============================================================================
# Simulation — OU velocity process
# ============================================================================

def simulate_phase(n_steps=150, dt=0.1, seed=42):
    """
    Simulate a 15-second phase of play using an Ornstein-Uhlenbeck
    velocity model for each player.

    Velocity follows:
        v_{k+1} = v_k + (-theta*(v_k - v_drift) + sigma_a*N(0,1)) * dt

    This is consistent with the DWNA process noise model Q in the
    Kalman filter — both assume velocity is driven by white-noise
    accelerations.  Role-based v_drift encodes tactical intent;
    sigma_a controls how energetically each player moves.

    Assumptions
    -----------
    - theta = 0.4 s^{-1} for all players (memory timescale ~2.5 s)
    - v_drift and sigma_a are role-dependent (see PLAYERS table)
    - Max speed capped at 9.8 m/s (~35 km/h)
    - Pitch boundaries enforced with soft velocity reflection
    """
    rng   = np.random.default_rng(seed)
    theta = 0.4
    max_v = 9.8

    all_pos = {}
    for name, team, x0, y0, vx_d, vy_d, sig in PLAYERS:
        pos = np.zeros((n_steps, 2))
        vel = np.array([vx_d * 0.3, vy_d * 0.3])  # start at fraction of drift
        pos[0] = [x0, y0]

        for i in range(1, n_steps):
            v_drift = np.array([vx_d, vy_d])
            noise   = rng.normal(0, 1, 2)
            vel     = vel + (-theta*(vel - v_drift) + sig*noise) * dt
            spd     = np.linalg.norm(vel)
            if spd > max_v:
                vel = vel * max_v / spd
            new_pos = pos[i-1] + vel * dt
            if new_pos[0] < 1  or new_pos[0] > 104: vel[0] *= -0.6
            if new_pos[1] < 1  or new_pos[1] > 67:  vel[1] *= -0.6
            pos[i] = np.clip(new_pos, [1, 1], [104, 67])

        all_pos[name] = pos

    return all_pos


def add_gps_noise(pos_dict, sigma, seed):
    """Zero-mean Gaussian GPS noise, independent per player and axis."""
    rng = np.random.default_rng(seed)
    return {n: p + rng.normal(0, sigma, p.shape) for n, p in pos_dict.items()}


# ============================================================================
# Covariance ellipse
# ============================================================================

def cov_ellipse(ax, mean, cov, n_std=2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    idx  = vals.argsort()[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    ang  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=ang, **kw))


# ============================================================================
# Figures
# ============================================================================

def plot_snapshot(pos_dict, meas_dict, filtered, snap, out_dir):
    """
    Figure 1: All 22 players at t = snap * dt seconds.

    GPS cloud, KF position estimate, velocity arrow (latent state),
    and 2-sigma covariance ellipse for each player.
    """
    fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
    ax  = fig.add_subplot(111)
    draw_pitch(ax)

    for name, team, *_ in PLAYERS:
        col = TEAM_COLORS[team]
        meas = meas_dict[name]
        xs   = filtered[name]['xs']
        Ps   = filtered[name]['Ps']

        # GPS cloud (last 30 steps)
        ax.scatter(*meas[max(0, snap-30):snap].T,
                   s=8, c=col, alpha=0.20, zorder=2)

        # KF trail
        ax.plot(*xs[max(0, snap-40):snap, :2].T,
                color=col, lw=1.0, alpha=0.45, zorder=3)

        # KF position + covariance ellipse
        pos_est = xs[snap, :2]
        cov_ellipse(ax, pos_est, Ps[snap, :2, :2],
                    alpha=0.18, color=col, zorder=4)
        ax.scatter(*pos_est, s=90, c=col,
                   edgecolors=WHITE, linewidths=0.8, zorder=5)

        # Velocity arrow (latent state — never observed)
        vel_est = xs[snap, 2:4]
        spd     = np.linalg.norm(vel_est)
        if spd > 0.5:
            scale = min(2.5, spd) / spd
            ax.annotate('', xy=pos_est + vel_est*scale*0.55,
                        xytext=pos_est,
                        arrowprops=dict(arrowstyle='->',
                                        color=col, lw=1.6,
                                        mutation_scale=10), zorder=6)

        # Name label
        ax.text(pos_est[0], pos_est[1] + 2.8, name[:7],
                ha='center', color=col,
                fontsize=6.5, fontweight='bold', zorder=7)

    legend_els = [
        Line2D([0],[0], marker='o', color='w', label='Norway (NOR)',
               markerfacecolor=NOR_RED,  markersize=9),
        Line2D([0],[0], marker='o', color='w', label='France (FRA)',
               markerfacecolor=FRA_BLUE, markersize=9),
        Line2D([0],[0], color=WHITE, lw=1.5,
               label='KF trajectory (L1)'),
        Line2D([0],[0], color='grey', lw=0, marker='.',
               markersize=6, alpha=0.5, label='GPS cloud (L0)'),
    ]
    ax.legend(handles=legend_els, loc='upper left', fontsize=9,
              facecolor=BG, edgecolor=WHITE, labelcolor=WHITE,
              framealpha=0.9)
    ax.set_title(
        f'22-Player Kalman Filter Tracking  |  t = {snap*0.1:.1f} s  |  '
        f'Norway (attacking) vs France (defending)  |  '
        f'FIFA World Cup 2026, June 26',
        color=WHITE, fontsize=10, pad=8)
    ax.text(52.5, -3.5,
            'Circles = KF position  |  '
            'Arrows = KF velocity (latent)  |  '
            'Ellipses = 2σ covariance',
            ha='center', color='#aaa', fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_match_snapshot.png'),
                dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('    Saved 01_match_snapshot.png')


def plot_speeds(filtered, snap, out_dir):
    """
    Figure 2: Player speeds at snapshot time, derived from KF velocity state.
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.set_facecolor('#1a2035')

    names, speeds, colors = [], [], []
    for name, team, *_ in PLAYERS:
        xs = filtered[name]['xs']
        v  = xs[snap, 2:4]
        names.append(name)
        speeds.append(np.linalg.norm(v) * 3.6)
        colors.append(TEAM_COLORS[team])

    ax.bar(names, speeds, color=colors, edgecolor=WHITE, linewidth=0.5)
    ax.axhline(24, color=GOLD, lw=1.2, ls='--', alpha=0.7)
    ax.text(0.5, 25, 'High-intensity threshold (24 km/h)',
            color=GOLD, fontsize=8)
    ax.set_title(
        f'Player Speeds at t = {snap*0.1:.1f} s  (from KF velocity state)  |  '
        f'NOR vs FRA  |  FIFA World Cup 2026',
        color=WHITE, fontsize=10)
    ax.set_ylabel('Speed [km/h]', color=WHITE)
    ax.tick_params(colors=WHITE, axis='y')
    ax.tick_params(colors=WHITE, axis='x', labelsize=7.5, rotation=45)
    for sp in ax.spines.values(): sp.set_color('#444')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_player_speeds.png'),
                dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('    Saved 02_player_speeds.png')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='22-Player Match Tracker — NOR vs FRA, World Cup 2026')
    parser.add_argument('--n-steps',   type=int,   default=150,
                        help='Simulation steps (default 150 = 15 s at 10 Hz)')
    parser.add_argument('--dt',        type=float, default=0.1)
    parser.add_argument('--sigma-gps', type=float, default=0.8)
    parser.add_argument('--sigma-q',   type=float, default=2.0)
    parser.add_argument('--snap',      type=int,   default=120,
                        help='Snapshot step for figures (default 120 = 12 s)')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--out',       type=str,   default='outputs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('=== 22-Player Match Tracker — NOR vs FRA, FIFA World Cup 2026 ===')
    print(f'    {args.n_steps} steps  |  dt={args.dt} s  |  '
          f'{args.n_steps * args.dt:.1f} s phase\n')

    # Simulate
    print('Simulating...')
    pos_dict  = simulate_phase(args.n_steps, args.dt, args.seed)
    meas_dict = add_gps_noise(pos_dict, args.sigma_gps, args.seed)

    # Filter
    print('Running Kalman filters...')
    kf       = KalmanFilter(args.dt, args.sigma_q, args.sigma_gps)
    filtered = {}
    rmse_vals = {}
    for name in pos_dict:
        xs, Ps = kf.run(meas_dict[name])
        filtered[name] = {'xs': xs, 'Ps': Ps}
        rmse_vals[name] = float(np.sqrt(
            ((xs[:, :2] - pos_dict[name])**2).mean()))

    raw_rmse  = float(np.mean([
        np.sqrt(((meas_dict[n] - pos_dict[n])**2).mean())
        for n in pos_dict]))
    mean_rmse = float(np.mean(list(rmse_vals.values())))
    print(f'    Raw GPS RMSE (mean) : {raw_rmse:.3f} m')
    print(f'    KF RMSE (mean)      : {mean_rmse:.3f} m  '
          f'({100*(1-mean_rmse/raw_rmse):.1f}% improvement)')
    print(f'    Haaland KF RMSE     : {rmse_vals["Haaland"]:.3f} m')

    # Figures
    snap = min(args.snap, args.n_steps - 1)
    print('\nGenerating figures...')
    plot_snapshot(pos_dict, meas_dict, filtered, snap, args.out)
    plot_speeds(filtered, snap, args.out)

    # Metrics
    metrics = {
        'match'            : 'Norway vs France — FIFA World Cup 2026, Group I',
        'date'             : 'June 26, 2026',
        'n_players'        : len(pos_dict),
        'phase_duration_s' : args.n_steps * args.dt,
        'gps_noise_sigma_m': args.sigma_gps,
        'raw_gps_rmse_m'   : round(raw_rmse,  4),
        'mean_kf_rmse_m'   : round(mean_rmse, 4),
        'improvement_pct'  : round(100*(1-mean_rmse/raw_rmse), 1),
        'per_player_rmse_m': {n: round(v, 4) for n, v in rmse_vals.items()},
    }
    mpath = os.path.join(args.out, 'match_metrics.json')
    with open(mpath, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n=== Summary ===')
    print(f'  {len(pos_dict)} players tracked simultaneously')
    print(f'  Mean KF RMSE : {mean_rmse:.3f} m  '
          f'({100*(1-mean_rmse/raw_rmse):.1f}% improvement)')
    print(f'  Metrics : {mpath}')
    print(f'  Figures : {args.out}/')


if __name__ == '__main__':
    main()
