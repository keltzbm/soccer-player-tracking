"""
simulate.py
-----------
Simulates GPS tracking data for 11 players in a 4-3-3 formation.

The team moves as a unit driven by a centroid trajectory. Each player
occupies a fixed formation offset from the centroid, plus role-based
Gaussian noise representing game unpredictability.

    pos_player(t) = pos_centroid(t) + offset + N(0, sigma)

The centroid follows a slow realistic path — advancing, holding shape,
dropping back — driven by its own low-noise random walk.

Noise magnitude by role
-----------------------
    GK     0.3 m  — near-stationary, very predictable
    CB     0.8 m  — disciplined shape
    LB/RB  1.5 m  — occasional forward runs
    CM     1.8 m  — box to box
    LW/RW  3.0 m  — wide attackers, reacting to ball
    ST     3.5 m  — movement off the ball, most unpredictable

Assumptions
-----------
- Pitch: 105 x 68 m (FIFA standard)
- Team attacks left to right (toward x = 105)
- Centroid starts at x = 40, y = 34 (own half, centre)
- 45 minutes at 10 Hz (27,000 steps)
- Noise is zero-mean, white, Gaussian, independent per player and axis
"""

import numpy as np

DT      = 0.1
N_STEPS = 45 * 60 * 10   # 45 min at 10 Hz

# Formation offsets from centroid  (dx, dy)
# weight: how much the player follows the centroid (0 = stays home, 1 = fully follows)
PLAYERS = [
    #  name   role    dx    dy   sigma  weight
    ('GK',  'GK',  -32,   0,   0.3,   0.0),  # stays in goal box

    ('LB',  'LB',  -12, -24,   1.5,   0.5),
    ('CB1', 'CB',  -14,  -8,   0.8,   0.4),
    ('CB2', 'CB',  -14,   8,   0.8,   0.4),
    ('RB',  'RB',  -12,  24,   1.5,   0.5),

    ('LCM', 'CM',    0, -16,   1.8,   0.8),
    ('CM',  'CM',    2,   0,   1.8,   0.8),
    ('RCM', 'CM',    0,  16,   1.8,   0.8),

    ('LW',  'LW',   22, -24,   3.0,   1.0),
    ('ST',  'ST',   24,   0,   3.5,   1.0),
    ('RW',  'RW',   22,  24,   3.0,   1.0),
]

PLAYER_SIG = {p[0]: p[4] for p in PLAYERS}
PLAYER_OFF = {p[0]: np.array([p[2], p[3]], dtype=float) for p in PLAYERS}
PLAYER_W   = {p[0]: p[5] for p in PLAYERS}


def simulate_centroid(n_steps, dt, seed):
    """
    Simulate the team centroid as a mean-reverting random walk.

    The centroid oscillates around a home position (x=50, y=34) —
    the centre of the pitch — representing the team shifting forward
    and backward as play develops, without systematically advancing.
    """
    rng      = np.random.default_rng(seed)
    pos      = np.zeros((n_steps, 2))
    pos[0]   = [50.0, 34.0]
    home     = np.array([50.0, 34.0])
    vel      = np.zeros(2)
    theta    = 0.02   # mean-reversion rate — slow drift back to home

    for i in range(1, n_steps):
        noise   = rng.normal(0, 0.6, 2)
        vel     = 0.97 * vel - theta * (pos[i-1] - home) + noise * 0.15
        new_pos = pos[i-1] + vel * dt
        new_pos[0] = np.clip(new_pos[0], 30, 72)
        new_pos[1] = np.clip(new_pos[1], 22, 46)
        pos[i]  = new_pos

    return pos


def simulate(n_steps=N_STEPS, dt=DT, seed=42):
    """
    Each player's position has two noise components:

        pos(t) = home + formation_offset + team_shift(t) + individual_dev(t)

    team_shift   : shared random walk — drives collective movement.
                   Same signal applied to all players, creating covariance.
    individual_dev: role-based OU process — each player's personal
                   unpredictability around the shared team motion.

    The centroid is computed from the player positions after the fact —
    it is not simulated directly.

    Returns
    -------
    positions : dict  name -> (n_steps, 2) true positions [m]
    t         : (n_steps,) time vector [s]
    """
    rng = np.random.default_rng(seed)

    # ── Shared team movement (random walk in velocity) ────────────────
    team_shift = np.zeros((n_steps, 2))
    team_vel   = np.zeros(2)
    home       = np.array([52.5, 34.0])   # pitch centre

    for i in range(1, n_steps):
        team_vel       = 0.97 * team_vel + rng.normal(0, 0.6, 2) * 0.15
        new_shift      = team_shift[i-1] + team_vel * dt
        # Keep team within a realistic band
        clipped_home   = home + new_shift
        clipped_home   = np.clip(clipped_home, [28, 20], [74, 48])
        team_shift[i]  = clipped_home - home

    # ── Individual OU deviations ──────────────────────────────────────
    alpha = 0.98
    positions = {}

    for name, role, dx, dy, sigma, weight in PLAYERS:
        offset     = np.array([dx, dy], dtype=float)
        home_pos   = home + offset
        sigma_step = sigma * np.sqrt(1 - alpha ** 2)

        dev = np.zeros((n_steps, 2))
        for i in range(1, n_steps):
            dev[i] = alpha * dev[i-1] + rng.normal(0, sigma_step, 2)

        pos = home_pos + team_shift + dev
        pos = np.clip(pos, [1, 1], [104, 67])
        positions[name] = pos

    t = np.arange(n_steps) * dt
    return positions, t


def add_gps_noise(positions, sigma=0.8, seed=0):
    """
    Add GPS sensor noise on top of true positions.

    Separate from the movement noise — this represents sensor inaccuracy,
    not game unpredictability.
    """
    rng = np.random.default_rng(seed)
    return {
        name: pos + rng.normal(0, sigma, pos.shape)
        for name, pos in positions.items()
    }


if __name__ == '__main__':
    positions, t = simulate()
    noisy    = add_gps_noise(positions)
    centroid = np.mean([positions[p[0]] for p in PLAYERS], axis=0)

    print(f'Players  : {len(PLAYERS)}')
    print(f'Steps    : {N_STEPS}  ({N_STEPS * DT / 60:.0f} min at {1/DT:.0f} Hz)')
    print(f'Centroid : start ({centroid[0,0]:.1f}, {centroid[0,1]:.1f})'
          f'  end ({centroid[-1,0]:.1f}, {centroid[-1,1]:.1f})')
    print()
    print(f'{"Name":<5}  {"Role":<4}  {"sigma":>5}  {"offset":>12}')
    print('-' * 36)
    for name, role, dx, dy, sigma, weight in PLAYERS:
        print(f'{name:<5}  {role:<4}  {sigma:>5.1f}  ({dx:+4.0f}, {dy:+4.0f})')
