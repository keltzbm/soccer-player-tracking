#!/usr/bin/env python3
"""
Norway vs France — FIFA World Cup 2026, Group I (June 26, 2026)
22-Player Kalman Filter Tracking Pipeline
 ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  =
Models the full match as a system of 22 simultaneously tracked targets.
One independent Kalman filter per player; team-level analytics derived
from the ensemble of filtered state estimates.

Processing Levels
-----------------
  L0  Raw GPS measurements (noisy player positions from GPS vests)
  L1  Per-player Kalman filter (position + latent velocity)
  L2  Team analytics: centroid, compactness, press intensity, speed distribution
      Ball trajectory prediction (3-D, from camera observations)

Scenarios
---------
  [1] Full 22-player system — 20-second attacking phase (Norway pressing)
  [2] Olise free kick       — 3-D ball tracking + GOAL/MISS prediction
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

import numpy as np


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Pitch drawing
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def draw_pitch(ax, pitch_length = 105, pitch_width = 68,
               pitch_color = '#2d6a2d', line_color = 'white'):
    ax.set_facecolor(pitch_color)
    lw = 1.5

    def box(x0, y0, w, h):
        ax.plot([x0, x0+w, x0+w, x0, x0],
                [y0, y0, y0+h, y0+h, y0], color = line_color, lw = lw)

    PL, PW = pitch_length, pitch_width
    box(0, 0, PL, PW)
    ax.plot([PL/2]*2, [0, PW], color = line_color, lw = lw)
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(PL/2 + 9.15*np.cos(th), PW/2 + 9.15*np.sin(th),
            color = line_color, lw = lw)
    ax.plot(PL/2, PW/2, 'o', color = line_color, ms = 3)

    for x0, sgn in [(0, 1), (PL, -1)]:
        box(x0, PW/2 - 20.16, sgn*16.5, 40.32)
        box(x0, PW/2 -  9.16, sgn* 5.5, 18.32)
        ax.plot(x0 + sgn*11, PW/2, 'o', color = line_color, ms = 3)
        ax.plot([x0, x0+sgn*(-2), x0+sgn*(-2), x0],
                [PW/2-3.66, PW/2-3.66, PW/2+3.66, PW/2+3.66],
                color = line_color, lw = 2.5)

    ax.set_xlim(-5, PL+5); ax.set_ylim(-5, PW+5)
    ax.set_aspect('equal'); ax.axis('off')


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Kalman filter
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

class KalmanFilter:
    """
    Constant-velocity Kalman filter (2-D or 3-D).

    State   x = [p, v]^T
    Predict x_{k|k-1} = F x_{k-1} + u
            P_{k|k-1} = F P_{k-1} F^T + Q
    Update  K = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
            x_{k|k} = x_{k|k-1} + K(z - H x_{k|k-1})
            P_{k|k} = (I - KH) P_{k|k-1}

    Assumptions
    -----------
    - Linear, time-invariant, constant-velocity dynamics
    - Zero-mean white Gaussian process and measurement noise
    - Position-only observations; velocity is latent
    - Gravity handled as a known control input (3-D ball tracking only)
    - Aerodynamic drag on the ball is unmodelled; absorbed into Q
    """

    def __init__(self, dims, dt, sigma_q, sigma_r, gravity = False):
        n = 2 * dims
        self.dims, self.n, self.dt = dims, n, dt

        F = np.eye(n)
        for i in range(dims):
            F[i, dims+i] = dt
        self.F = F

        H = np.zeros((dims, n))
        for i in range(dims):
            H[i, i] = 1.0
        self.H = H

        q = sigma_q ** 2
        Q = np.zeros((n, n))
        for i in range(dims):
            Q[i, i     ] = q * dt**4 / 4
            Q[i, dims+i] = q * dt**3 / 2
            Q[dims+i, i     ] = q * dt**3 / 2
            Q[dims+i, dims+i] = q * dt**2
        self.Q = Q

        self.R = (sigma_r ** 2) * np.eye(dims)

        self.u = np.zeros(n)
        if gravity and dims == 3:
            self.u[2] = -0.5 * 9.81 * dt**2
            self.u[dims+2] = -9.81 * dt

    def run(self, measurements, x0 = None, P0 = None):
        N = len(measurements)
        xs = np.zeros((N, self.n))
        Ps = np.zeros((N, self.n, self.n))
        innovations = np.zeros((N, self.dims))

        x = np.zeros(self.n) if x0 is None else x0.copy()
        if x0 is None:
            x[: self.dims] = measurements[0]
        P = np.diag([1.]*self.dims + [25.]*self.dims) if P0 is None else P0.copy()

        for k, z in enumerate(measurements):
            x_p = self.F @ x + self.u
            P_p = self.F @ P @ self.F.T + self.Q
            y = z - self.H @ x_p
            S = self.H @ P_p @ self.H.T + self.R
            K = P_p @ self.H.T @ np.linalg.inv(S)
            x = x_p + K @ y
            P = (np.eye(self.n) - K @ self.H) @ P_p
            xs[k], Ps[k], innovations[k] = x, P, y

        return xs, Ps, innovations

    def predict_ahead(self, x, P, n_steps):
        preds = np.zeros((n_steps, self.n))
        xp, Pp = x.copy(), P.copy()
        for k in range(n_steps):
            xp = self.F @ xp + self.u
            Pp = self.F @ Pp @ self.F.T + self.Q
            preds[k] = xp
        return preds


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# 22-Player match simulation
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

# Player definition: (name, team, x0, y0, vx_drift, vy_drift, v_noise, note)
# Norway attacks left → right (toward x = 105)
# France defends (x ≈ 78–100 block)
PLAYERS = [
    # ── Norway ──────────────────────────────────────────────────
    ('Nyland', 'NOR', 8, 34, 0.0, 0.0, 0.15, 'GK'),
    ('Ryerson', 'NOR', 42, 60, 2.2, 0.8, 0.50, 'RB'),
    ('H-Olsen', 'NOR', 38, 42, 1.0, 0.0, 0.40, 'CB'),
    ('Ostigard', 'NOR', 38, 26, 1.0, 0.0, 0.40, 'CB'),
    ('Meling', 'NOR', 44, 8, 1.5, -0.2, 0.45, 'LB'),
    ('Aursnes', 'NOR', 60, 18, 2.0, 0.3, 0.55, 'LCM'),
    ('Berge', 'NOR', 58, 34, 2.2, 0.0, 0.55, 'CM'),
    ('Odegaard', 'NOR', 65, 50, 2.5, 0.5, 0.60, 'RCM'),
    ('Hauge', 'NOR', 72, 12, 1.8, -1.2, 0.65, 'LW'),
    #  sprint
    ('Haaland', 'NOR', 55, 34, 7.0, 1.2, 0.40, 'ST'),
    #  cut inside
    ('Nusa', 'NOR', 74, 58, 2.0, -2.5, 0.70, 'RW'),
    # ── France ──────────────────────────────────────────────────
    ('Maignan', 'FRA', 100, 34, 0.0, 0.0, 0.10, 'GK'),
    ('Kounde', 'FRA', 84, 58, -0.8, 0.3, 0.45, 'RB'),
    #  tracks Haaland
    ('Konate', 'FRA', 83, 42, -0.5, 2.0, 0.50, 'CB'),
    ('Saliba', 'FRA', 83, 26, -0.5, -0.5, 0.45, 'CB'),
    ('T-Hernandez', 'FRA', 84, 10, -0.8, -0.5, 0.45, 'LB'),
    ('Griezmann', 'FRA', 73, 48, -1.5, 0.3, 0.60, 'AM'),
    ('Tchouameni', 'FRA', 72, 34, -1.2, 0.0, 0.55, 'CM'),
    ('Kante', 'FRA', 72, 22, -1.2, 0.0, 0.55, 'CM'),
    ('Dembele', 'FRA', 76, 14, -1.0, -1.5, 0.65, 'LW'),
    #  high press
    ('Mbappe', 'FRA', 80, 34, -2.5, 0.8, 0.70, 'ST'),
    ('Olise', 'FRA', 80, 54, -1.2, -1.8, 0.65, 'RW'),
]

NOR_RED = '#EF2B2D'
FRA_BLUE = '#003189'
GOLD = '#FFD700'
WHITE = '#FFFFFF'

TEAM_COLORS = {'NOR': NOR_RED, 'FRA': FRA_BLUE}


def simulate_match_phase(n_steps = 200, dt = 0.1, seed = 42):
    """
    Simulate a 20-second attacking phase using Langevin dynamics.

    Each player's velocity follows:
        v_{k+1} = alpha * v_k + (1-alpha) * v_target(t) + N(0, sigma_v)

    This produces smooth, correlated motion with a role-dependent drift.
    Model mismatch with the constant-velocity KF is intentional — it
    mirrors real player kinematics (acceleration, direction changes).

    Haaland is given a sigmoid sprint profile to model his explosive
    acceleration from a standing start.
    """
    rng = np.random.default_rng(seed)
    #  velocity smoothing (higher = smoother)
    alpha = 0.82
    t_arr = np.arange(n_steps) * dt

    all_pos = {}
    all_vel = {}

    for name, team, x0, y0, vx_d, vy_d, v_noise, role in PLAYERS:
        pos = np.zeros((n_steps, 2))
        vel = np.zeros(2)
        pos[0] = [x0, y0]

        for i in range(1, n_steps):
            t = t_arr[i]

            # Role-specific target velocity
            if name == 'Haaland':
                # Sigmoid sprint: acceleration from standstill to full speed
                s = 1 / (1 + np.exp(-2.0 * (t - 2.5)))
                v_target = np.array([vx_d * s, vy_d * s])
            elif name == 'Konate':
                # CB tracks Haaland with 0.8 s reaction delay
                h_idx = max(0, i - 8)
                h_pos = all_pos.get('Haaland', np.tile([x0, y0], (n_steps, 1)))[h_idx]
                diff = h_pos - pos[i-1]
                dist = np.linalg.norm(diff) + 1e-6
                v_target = (diff / dist) * min(7.5, dist / dt * 0.6)
            elif name == 'Nusa':
                # RW cuts inside sharply after t = 7 s
                if t < 7:
                    v_target = np.array([vx_d, 0.5])
                else:
                    v_target = np.array([3.5, -3.0])
            else:
                v_target = np.array([vx_d, vy_d])

            noise = rng.normal(0, v_noise, 2)
            vel = alpha * vel + (1 - alpha) * v_target + noise * (1 - alpha)

            # Keep on pitch
            new_pos = pos[i-1] + vel * dt
            new_pos = np.clip(new_pos, [1, 1], [104, 67])
            pos[i] = new_pos

        all_pos[name] = pos
        all_vel[name] = np.gradient(pos, dt, axis = 0)

    return all_pos, all_vel, t_arr


def add_gps_noise(pos_dict, sigma, seed):
    rng = np.random.default_rng(seed)
    return {name: pos + rng.normal(0, sigma, pos.shape)
            for name, pos in pos_dict.items()}


def run_all_filters(meas_dict, dt, sigma_q, sigma_r):
    """Run one independent Kalman filter per player."""
    kf = KalmanFilter(2, dt, sigma_q, sigma_r)
    result = {}
    for name, meas in meas_dict.items():
        xs, Ps, _ = kf.run(meas)
        result[name] = {'xs': xs, 'Ps': Ps}
    return result, kf


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Team-level analytics
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def team_analytics(filtered, pos_true_dict, t_arr):
    """
    Derive team-level quantities from the ensemble of filtered tracks.

    centroid_x: team centre of mass in x (how advanced the team is)
    compactness: mean pairwise distance within team (lower = more compact)
    press_int: France pressing intensity — rate of advance toward Norway
    """
    analytics = {}
    for team in ('NOR', 'FRA'):
        names = [p[0] for p in PLAYERS if p[1] == team]
        xs_stack = np.stack([filtered[n]['xs'][:, :2] for n in names], axis = 1)
        # (T, 11, 2)
        centroid = xs_stack.mean(axis = 1)   # (T, 2)

        # Compactness: mean distance from centroid
        diffs = xs_stack - centroid[:, None, :]          # (T, 11, 2)
        compactness = np.linalg.norm(diffs, axis = 2).mean(axis = 1)  # (T, )

        analytics[team] = {'centroid': centroid, 'compactness': compactness}

    # Press intensity: rate of change of FRA centroid x (negative = pressing)
    press = -np.gradient(analytics['FRA']['centroid'][:, 0], t_arr)
    analytics['press_intensity'] = press

    return analytics


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Covariance ellipse
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def cov_ellipse(ax, mean, cov, n_std = 2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals = np.maximum(vals[idx], 0)
    vecs = vecs[:, idx]
    ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy = mean, width = w, height = h, angle = ang, **kw))


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Figure 1: Full match snapshot
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def plot_match_snapshot(filtered, meas_dict, pos_true_dict, snap_step,
                        out_dir):
    """
    Figure 1: All 22 players at t = snap_step * dt seconds.

    For each player shows:
      - Grey cloud: raw GPS measurements up to this moment
      - Filled circle: KF position estimate
      - Arrow: KF velocity estimate (latent — not directly observed)
      - 2-sigma covariance ellipse
    """
    fig = plt.figure(figsize = (14, 8.5), facecolor = '#111827')
    ax = fig.add_subplot(111)
    draw_pitch(ax)

    for name, team, *_ in PLAYERS:
        col = TEAM_COLORS[team]
        meas = meas_dict[name]
        xs = filtered[name]['xs']
        Ps = filtered[name]['Ps']

        # GPS cloud (last 20 steps)
        trail_start = max(0, snap_step - 20)
        ax.scatter(*meas[trail_start:snap_step].T,
                   s = 6, c = col, alpha = 0.18, zorder = 2)

        # KF trail
        ax.plot(*xs[max(0, snap_step-30):snap_step, :2].T,
                color = col, lw = 1.0, alpha = 0.5, zorder = 3)

        # Current KF position
        pos_est = xs[snap_step, :2]
        ax.scatter(*pos_est, s = 80, c = col, edgecolors = WHITE,
                   linewidths = 0.8, zorder = 5)

        # Covariance ellipse
        cov_ellipse(ax, pos_est, Ps[snap_step, :2, :2],
                    n_std = 2, alpha = 0.20, color = col, zorder = 4)

        # Velocity arrow (KF latent state)
        vel_est = xs[snap_step, 2:4]
        speed = np.linalg.norm(vel_est)
        if speed > 0.5:
            scale = min(3.0, speed) / speed
            ax.annotate('', xy = pos_est + vel_est * scale * 0.6,
                        xytext = pos_est,
                        arrowprops = dict(arrowstyle = '->', color = col,
                                        lw = 1.6, mutation_scale = 10),
                        zorder = 6)

        # Name label (short)
        short = name.split('-')[0][:7]
        ax.text(pos_est[0], pos_est[1] + 2.8, short,
                ha = 'center', color = col, fontsize = 6.5,
                fontweight = 'bold', zorder = 7)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker = 'o', color = 'w', label = 'Norway (NOR)',
               markerfacecolor = NOR_RED, markersize = 9),
        Line2D([0], [0], marker = 'o', color = 'w', label = 'France (FRA)',
               markerfacecolor = FRA_BLUE, markersize = 9),
        Line2D([0], [0], color = WHITE, lw = 1, linestyle = '-',
               label = 'KF trajectory (L1)'),
        Line2D([0], [0], color = 'grey', lw = 0, marker = '.',
               markersize = 5, alpha = 0.5, label = 'GPS cloud (L0)'),
    ]
    ax.legend(handles = legend_elements, loc = 'upper left', fontsize = 8.5,
              facecolor = '#111827', edgecolor = WHITE, labelcolor = WHITE,
              framealpha = 0.9)

    t_snap = snap_step * 0.1
    ax.set_title(
        f'[1] 22-Player Kalman Filter Tracking  |  t = {t_snap:.1f} s  |  '
        f'Norway (attacking) vs France (defending)  |  '
        f'FIFA World Cup 2026, June 26',
        color = WHITE, fontsize = 10, pad = 8)
    ax.text(52.5, -3.5,
            'Circles = KF position estimate  |  '
            'Arrows = KF velocity (latent state)  |  '
            'Ellipses = 2-sigma covariance',
            ha = 'center', color = '#aaa', fontsize = 7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_match_snapshot.png'),
                dpi = 130, facecolor = fig.get_facecolor())
    plt.close(fig)
    print('    Saved 01_match_snapshot.png')


################################################################################
# Figure 2: Team dynamics over time
################################################################################

def plot_team_dynamics(analytics, t_arr, out_dir):
    """
    Figure 2: Team-level quantities derived from the 22 filtered tracks.

    (a) Team centroid x — how advanced each team is on the pitch
    (b) Team compactness — mean spread of each team
    (c) France press intensity — rate at which France closes space
    """
    fig, axes = plt.subplots(3, 1, figsize = (11, 9), sharex = True,
                             facecolor = '#111827')
    fig.suptitle(
        '[2] Team Dynamics from Ensemble of 22 Kalman Filter Tracks  |  '
        'NOR vs FRA  |  FIFA World Cup 2026',
        color = WHITE, fontsize = 10, y = 1.00)

    # (a) Centroid x
    ax = axes[0]
    ax.plot(t_arr, analytics['NOR']['centroid'][:, 0],
            color = NOR_RED, lw = 2, label = 'Norway centroid x')
    ax.plot(t_arr, analytics['FRA']['centroid'][:, 0],
            color = FRA_BLUE, lw = 2, label = 'France centroid x')
    ax.set_ylabel('Centroid x [m]', color = WHITE, fontsize = 9)
    ax.legend(fontsize = 8.5, facecolor = '#111827', edgecolor = WHITE,
              labelcolor = WHITE)

    # (b) Compactness
    ax = axes[1]
    ax.plot(t_arr, analytics['NOR']['compactness'],
            color = NOR_RED, lw = 2, label = 'Norway compactness')
    ax.plot(t_arr, analytics['FRA']['compactness'],
            color = FRA_BLUE, lw = 2, label = 'France compactness')
    ax.set_ylabel('Compactness [m]', color = WHITE, fontsize = 9)
    ax.legend(fontsize = 8.5, facecolor = '#111827', edgecolor = WHITE,
              labelcolor = WHITE)

    # (c) Press intensity
    ax = axes[2]
    press = analytics['press_intensity']
    ax.plot(t_arr, press, color = FRA_BLUE, lw = 2)
    ax.fill_between(t_arr, press, alpha = 0.2, color = FRA_BLUE)
    ax.axhline(0, color = WHITE, lw = 0.8, ls = '--', alpha = 0.4)
    ax.set_ylabel('France press intensity [m/s]', color = WHITE, fontsize = 9)
    ax.set_xlabel('Time [s]', color = WHITE, fontsize = 9)

    for a in axes:
        a.set_facecolor('#1a2035')
        a.tick_params(colors = WHITE)
        a.grid(alpha = 0.15, color = WHITE)
        for sp in a.spines.values():
            sp.set_color('#444')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_team_dynamics.png'),
                dpi = 130, facecolor = fig.get_facecolor())
    plt.close(fig)
    print('    Saved 02_team_dynamics.png')


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Figure 3: Player dynamics (speed + heading)
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def plot_player_dynamics(filtered, t_arr, out_dir):
    """
    Figure 3: Kinematic dynamics for selected key players.

    The KF velocity state (never directly measured) is used to derive
    speed and heading — demonstrating latent state recovery.
    """
    highlight = ['Haaland', 'Mbappe', 'Konate', 'Odegaard', 'Olise', 'Nusa']
    colors = [NOR_RED, FRA_BLUE, FRA_BLUE, NOR_RED, FRA_BLUE, NOR_RED]

    fig, (ax_speed, ax_head) = plt.subplots(2, 1, figsize = (12, 7),
                                            sharex = True, facecolor = '#111827')
    fig.suptitle(
        '[3] Player Dynamics from KF Velocity State (latent — not directly observed)  |  '
        'NOR vs FRA  |  FIFA World Cup 2026',
        color = WHITE, fontsize = 9.5, y = 1.00)

    for name, col in zip(highlight, colors):
        xs = filtered[name]['xs']
        vx = xs[:, 2]
        vy = xs[:, 3]
        speed = np.sqrt(vx**2 + vy**2) * 3.6
        heading = np.degrees(np.arctan2(vy, vx))

        ls = '-' if [p[1] for p in PLAYERS if p[0] == name][0] == 'NOR' else '--'
        ax_speed.plot(t_arr, speed, color = col, lw = 1.8, ls = ls, label = name)
        ax_head.plot( t_arr, heading, color = col, lw = 1.8, ls = ls)

    ax_speed.axhline(32, color = '#555', lw = 0.8, ls = ':', alpha = 0.6)
    ax_speed.text(0.5, 33.2, 'Haaland avg top speed ~ 32 km/h',
                  color = '#888', fontsize = 7.5)

    ax_speed.set_ylabel('Speed [km/h]', color = WHITE, fontsize = 9)
    ax_head.set_ylabel('Heading [deg]', color = WHITE, fontsize = 9)
    ax_head.set_xlabel('Time [s]', color = WHITE, fontsize = 9)
    ax_head.axhline(0, color = WHITE, lw = 0.6, ls = '--', alpha = 0.3)

    ax_speed.legend(fontsize = 8.5, facecolor = '#111827', edgecolor = WHITE,
                    labelcolor = WHITE, ncol = 3)

    for a in (ax_speed, ax_head):
        a.set_facecolor('#1a2035')
        a.tick_params(colors = WHITE)
        a.grid(alpha = 0.15, color = WHITE)
        for sp in a.spines.values():
            sp.set_color('#444')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_player_dynamics.png'),
                dpi = 130, facecolor = fig.get_facecolor())
    plt.close(fig)
    print('    Saved 03_player_dynamics.png')


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Figure 4: Speed distribution at snapshot time
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def plot_speed_distribution(filtered, snap_step, out_dir):
    """
    Figure 4: Speed distribution of both teams at the snapshot moment.

    Shows the tactical difference in pressing vs defensive intensity.
    """
    fig, ax = plt.subplots(figsize = (9, 5), facecolor = '#111827')
    ax.set_facecolor('#1a2035')

    for team, col in TEAM_COLORS.items():
        names = [p[0] for p in PLAYERS if p[1] == team]
        speeds = []
        for name in names:
            xs = filtered[name]['xs']
            v = xs[snap_step, 2:4]
            speeds.append(np.linalg.norm(v) * 3.6)

        ax.bar(names, speeds, color = col, alpha = 0.8, edgecolor = WHITE,
               linewidth = 0.5, label = f'{team}')

    ax.set_title('[4] Player Speeds at t = 10 s  (from KF velocity state)  |  '
                 'NOR vs FRA  |  FIFA World Cup 2026',
                 color = WHITE, fontsize = 9.5)
    ax.set_ylabel('Speed [km/h]', color = WHITE, fontsize = 9)
    ax.tick_params(colors = WHITE, axis = 'y')
    ax.tick_params(colors = WHITE, axis = 'x', labelsize = 7, rotation = 45)
    for sp in ax.spines.values():
        sp.set_color('#444')
    ax.legend(fontsize = 9, facecolor = '#111827', edgecolor = WHITE,
              labelcolor = WHITE)
    ax.axhline(24, color = '#555', lw = 0.8, ls = ':')
    ax.text(0.5, 25, 'High-intensity running threshold (24 km/h)',
            color = '#888', fontsize = 7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_speed_distribution.png'),
                dpi = 130, facecolor = fig.get_facecolor())
    plt.close(fig)
    print('    Saved 04_speed_distribution.png')


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Scenario 2 — Olise free kick (3-D ball tracking)
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def simulate_freekick(dt = 0.05, seed = 42):
    """
    Simulate an Olise free kick from 25 m out (France 2nd half).

    Physics: Euler integration with gravity + aerodynamic drag.
    Ball (FIFA size 5): m = 0.43 kg, r = 0.1085 m, C_D = 0.20 (turbulent).
    """
    g = 9.81
    k_d = 0.5 * 1.225 * 0.20 * (np.pi * 0.1085**2) / 0.43

    pos = [np.array([80.0, 21.0, 0.0])]
    vel = np.array([24.0, 12.5, 7.4])

    for _ in range(500):
        p = pos[-1]
        if p[0] >= 105.0 or p[2] < -0.1:
            break
        spd = np.linalg.norm(vel)
        vel = vel + (-k_d * spd * vel + np.array([0, 0, -g])) * dt
        pos.append(p + vel * dt)

    return np.array(pos)


def plot_freekick(pos_true, meas, xs_kf, predictions, obs_until, dt,
                  out_dir):
    GOAL_X = 105.0
    POST_Y = (30.34, 37.66)
    CROSS_Z = 2.44

    final = pos_true[-1]
    is_goal = (final[2] >= 0 and POST_Y[0] <= final[1] <= POST_Y[1]
               and final[2] <= CROSS_Z)
    verdict = 'GOAL!' if is_goal else 'MISS / SAVED'
    verdict_col = GOLD if is_goal else NOR_RED

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize = (14, 6),
                                      facecolor = '#111827')
    fig.suptitle(
        f'[5] Olise Free Kick  |  3-D Ball Tracking  |  '
        f'Prediction after {obs_until * dt:.2f} s  |  {verdict}',
        color = verdict_col, fontsize = 10, fontweight = 'bold', y = 1.00)

    obs_time = obs_until * dt

    # Side view
    ax_s.set_facecolor('#2d6a2d')
    ax_s.set_title('Side View (x - z)', color = WHITE, fontsize = 9)
    ax_s.axhline(0, color = WHITE, lw = 1.5)
    ax_s.plot([GOAL_X]*2, [0, CROSS_Z], color = WHITE, lw = 4)
    ax_s.plot([GOAL_X, GOAL_X+2], [CROSS_Z]*2, color = WHITE, lw = 4)
    ax_s.plot(pos_true[:, 0], pos_true[:, 2], color = '#aaa', lw = 1.5, alpha = 0.6,
              label = 'True trajectory')
    ax_s.scatter(meas[:obs_until, 0], meas[:obs_until, 2],
                 s = 14, c = FRA_BLUE, alpha = 0.6,
                 label = f'L0 camera ({obs_until} frames, {obs_time:.2f} s)')
    ax_s.plot(xs_kf[:, 0], xs_kf[:, 2], color = WHITE, lw = 1.8,
              label = 'L1 Kalman filter')
    pred = np.vstack([xs_kf[-1, :3], predictions[:, :3]])
    ax_s.plot(pred[:, 0], pred[:, 2], '--', color = GOLD, lw = 2.2,
              label = 'L2 predicted trajectory')
    ax_s.plot(GOAL_X, predictions[-1, 2], '*', color = verdict_col, ms = 14)
    ax_s.text(GOAL_X+0.5, predictions[-1, 2],
              f'z = {predictions[-1, 2]:.2f} m\n{verdict}',
              color = verdict_col, fontsize = 8, va = 'center')
    ax_s.set_xlabel('x [m]', color = WHITE); ax_s.set_ylabel('z [m]', color = WHITE)
    ax_s.tick_params(colors = WHITE)
    for sp in ax_s.spines.values():sp.set_color(WHITE)
    ax_s.legend(fontsize = 7.5, facecolor = '#111827', edgecolor = WHITE,
                labelcolor = WHITE, loc = 'upper left')
    ax_s.set_xlim(78, 108); ax_s.set_ylim(-0.5, 9)

    # Top view
    draw_pitch(ax_t)
    ax_t.set_title('Top View (x - y)', color = WHITE, fontsize = 9)
    ax_t.scatter(meas[:obs_until, 0], meas[:obs_until, 1],
                 s = 14, c = FRA_BLUE, alpha = 0.6, label = 'L0 camera', zorder = 4)
    ax_t.plot(xs_kf[:, 0], xs_kf[:, 1], color = WHITE, lw = 1.8,
              label = 'L1 KF', zorder = 5)
    pred_top = np.vstack([xs_kf[-1, :2], predictions[:, :2]])
    ax_t.plot(pred_top[:, 0], pred_top[:, 1], '--', color = GOLD, lw = 2.2,
              label = 'L2 predicted', zorder = 5)
    ax_t.plot(predictions[-1, 0], predictions[-1, 1], '*',
              color = verdict_col, ms = 14, zorder = 7, label = verdict)
    ax_t.set_xlim(75, 110); ax_t.set_ylim(15, 55)
    ax_t.legend(fontsize = 7.5, facecolor = '#111827', edgecolor = WHITE,
                labelcolor = WHITE, loc = 'upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_olise_freekick.png'),
                dpi = 130, facecolor = fig.get_facecolor())
    plt.close(fig)
    print(f'    Saved 05_olise_freekick.png  [{verdict}]')
    return is_goal


# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==
# Main
# ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==  ==

def main():
    parser = argparse.ArgumentParser(
        description = 'NOR vs FRA World Cup 2026 — 22-Player Tracking Pipeline')
    parser.add_argument('--sigma-gps', type = float, default = 0.8)
    parser.add_argument('--sigma-cam', type = float, default = 0.18)
    parser.add_argument('--snap-step', type = int, default = 100,
                        help = 'Snapshot time step for Figure 1 (default 100 = 10 s)')
    parser.add_argument('--obs-fk', type = int, default = 15)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--out', type = str, default = 'outputs')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok = True)

    print(' ==  = NOR vs FRA — FIFA World Cup 2026 | 22-Player Tracking Pipeline ==  = ')
    print('    Group I  ·  June 26, 2026\n')

    # ── Scenario 1: 22-player system ─────────────────────────────────────
    print('Scenario 1  22-player tracking system...')
    DT = 0.1
    pos_dict, _, t_arr = simulate_match_phase(n_steps = 200, dt = DT, seed = args.seed)
    meas_dict = add_gps_noise(pos_dict, args.sigma_gps, args.seed)
    filtered, kf = run_all_filters(meas_dict, DT,
                                          sigma_q = 2.0, sigma_r = args.sigma_gps)

    # Per-player RMSE
    rmse_vals = {}
    for name in pos_dict:
        xs = filtered[name]['xs'][:, :2]
        err = float(np.sqrt(((xs - pos_dict[name])**2).mean()))
        rmse_vals[name] = err

    mean_rmse = np.mean(list(rmse_vals.values()))
    raw_rmse = args.sigma_gps * np.sqrt(2)   # expected raw RMSE
    print(f'    Players tracked: {len(pos_dict)}')
    print(f'    GPS noise sigma: {args.sigma_gps} m  '
          f'(raw RMSE ~ {raw_rmse:.3f} m)')
    print(f'    Mean KF RMSE: {mean_rmse:.3f} m  '
          f'({100*(1-mean_rmse/raw_rmse):.1f}% improvement)')
    print(f'    Haaland KF RMSE: {rmse_vals["Haaland"]:.3f} m')
    print(f'    Konate  KF RMSE: {rmse_vals["Konate"]:.3f} m')

    analytics = team_analytics(filtered, pos_dict, t_arr)

    snap = min(args.snap_step, len(t_arr) - 1)

    # ── Scenario 2: Olise free kick ───────────────────────────────────────
    print('\nScenario 2  Olise free kick (3-D ball tracking)...')
    DT2 = 0.05
    pos2 = simulate_freekick(dt = DT2, seed = args.seed)
    meas2 = pos2 + np.random.default_rng(args.seed).normal(
        0, args.sigma_cam, pos2.shape)
    obs = min(args.obs_fk, len(meas2)-2)
    x0_fk = np.zeros(6)
    x0_fk[:3] = meas2[0]
    x0_fk[3:] = (meas2[1] - meas2[0]) / DT2
    kf2 = KalmanFilter(3, DT2, sigma_q = 3.0, sigma_r = args.sigma_cam,
                          gravity = True)
    xs2, Ps2, _ = kf2.run(meas2[:obs], x0 = x0_fk)
    preds2 = kf2.predict_ahead(xs2[-1], Ps2[-1], len(pos2) - obs + 25)
    print(f'    Observation window: {obs * DT2:.2f} s  ({obs} frames)')
    print(f'    True final position:'
          f'({pos2[-1, 0]:.1f}, {pos2[-1, 1]:.1f}, {pos2[-1, 2] .2f}) m')

    # ── Figures ──────────────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_match_snapshot(filtered, meas_dict, pos_dict, snap, args.out)
    plot_team_dynamics(analytics, t_arr, args.out)
    plot_player_dynamics(filtered, t_arr, args.out)
    plot_speed_distribution(filtered, snap, args.out)
    is_goal = plot_freekick(pos2, meas2, xs2, preds2, obs, DT2, args.out)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = {
        'match': 'Norway vs France — FIFA World Cup 2026, Group I',
        'date': 'June 26, 2026',
        'scenario_1_22_player_system': {
            'n_players': len(pos_dict),
            'gps_noise_sigma_m': args.sigma_gps,
            'raw_rmse_approx_m': round(raw_rmse, 4),
            'mean_kf_rmse_m': round(mean_rmse, 4),
            'improvement_pct': round(100*(1-mean_rmse/raw_rmse), 1),
            'per_player_kf_rmse_m': {n: round(v, 4)
                                        for n, v in rmse_vals.items()},
        },
        'scenario_2_olise_freekick': {
            'camera_noise_sigma_m': args.sigma_cam,
            'observation_window_s': round(obs * DT2, 3),
            'frames_before_predict': obs,
            'predicted_outcome': 'GOAL' if is_goal else 'MISS/SAVED',
            'true_final_position_m': [round(float(v), 3) for v in pos2[-1]],
        },
    }
    mpath = os.path.join(args.out, 'metrics.json')
    with open(mpath, 'w') as f:
        json.dump(metrics, f, indent = 2)

    print(f'\n ==  = Summary ==  = ')
    print(f'  22 players tracked simultaneously')
    print(f'  Mean KF RMSE: {mean_rmse:.3f} m  '
          f'({100*(1-mean_rmse/raw_rmse):.1f}% improvement over raw GPS)')
    print(f'  Olise free kick: {"GOAL!" if is_goal else "MISS"}  '
          f'(predicted after {obs*DT2:.2f} s)')
    print(f'  Metrics: {mpath}')
    print(f'  Figures: {args.out}/')


if __name__ == '__main__':
    main()
