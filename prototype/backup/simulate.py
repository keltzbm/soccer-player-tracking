"""
simulate.py
-----------
Simulates 22-player GPS tracking data with a possession model.

Norway vs France — FIFA World Cup 2026, Group I, June 26

Movement
--------
Each player moves at a constant role-based velocity plus Gaussian jitter:

    pos_{k+1} = pos_k + (vel + N(0, sigma_a)) * dt

When a team loses possession their players press toward the ball.

Possession Model
----------------
At each step, with probability p_event a pass is attempted:

  1. Voronoi regions are computed for both teams — each player "owns"
     the pitch area closer to them than any teammate.
  2. The possessor samples a target teammate weighted by Voronoi area
     (more space = better passing option).
  3. Pass success probability depends on distance and the number of
     defenders in the passing lane.
  4. If the pass succeeds, possession transfers along that edge.
  5. If it fails (error), the nearest opponent intercepts.

The randomness in step 3 models everything from a clean interception
to a dribbler beating a press — no determinism.

Assumptions
-----------
- Pitch: 105 x 68 m (FIFA standard)
- Norway attacks left to right (toward x = 105)
- Norway starts with possession
- p_event = 0.12 per step (~1 pass attempt per 0.8 s)
- Voronoi areas computed on a 2 m grid
- Pressing speed when out of possession: 4 m/s
"""

import numpy as np

DT      = 0.1              # time step [s]
N_STEPS = 45 * 60 * 10    # 45 minutes at 10 Hz

# (name, team, x0, y0, vx, vy, sigma_a)
PLAYERS = [
    ('Nyland',      'NOR',  8, 34,  0.0,  0.0, 0.5),
    ('Ryerson',     'NOR', 42, 58,  2.2,  0.8, 2.0),
    ('H-Olsen',     'NOR', 38, 42,  1.0,  0.0, 1.5),
    ('Ostigard',    'NOR', 38, 26,  1.0,  0.0, 1.5),
    ('Meling',      'NOR', 44, 10,  1.8, -0.5, 2.0),
    ('Aursnes',     'NOR', 60, 18,  2.2,  0.3, 2.5),
    ('Berge',       'NOR', 58, 34,  2.5,  0.0, 2.5),
    ('Odegaard',    'NOR', 65, 50,  2.8,  0.5, 2.5),
    ('Hauge',       'NOR', 72, 12,  2.0, -1.5, 3.0),
    ('Haaland',     'NOR', 55, 34,  6.0,  1.0, 3.5),
    ('Nusa',        'NOR', 74, 58,  2.5, -3.0, 3.0),
    ('Maignan',     'FRA',100, 34,  0.0,  0.0, 0.5),
    ('Kounde',      'FRA', 84, 58, -0.8,  0.3, 2.0),
    ('Konate',      'FRA', 83, 42, -0.5,  2.0, 2.0),
    ('Saliba',      'FRA', 83, 26, -0.5, -0.5, 2.0),
    ('T-Hernandez', 'FRA', 84, 10, -0.8, -0.5, 2.0),
    ('Griezmann',   'FRA', 73, 48, -1.5,  0.3, 2.5),
    ('Tchouameni',  'FRA', 72, 34, -1.2,  0.0, 2.5),
    ('Kante',       'FRA', 72, 22, -1.2,  0.0, 2.5),
    ('Dembele',     'FRA', 76, 14, -1.0, -1.5, 3.0),
    ('Mbappe',      'FRA', 80, 34, -2.8,  0.8, 3.0),
    ('Olise',       'FRA', 80, 54, -1.2, -2.0, 3.0),
]

# Lookup helpers
PLAYER_TEAM = {p[0]: p[1] for p in PLAYERS}
PLAYER_SIG  = {p[0]: p[6] for p in PLAYERS}

# Attack velocities — used when a team has possession
# NOR attacks toward x=105 (positive x), FRA toward x=0 (negative x)
ATTACK_VEL = {
    'Nyland':       np.array([ 0.0,  0.0]),
    'Ryerson':      np.array([ 2.2,  0.8]),
    'H-Olsen':      np.array([ 1.0,  0.0]),
    'Ostigard':     np.array([ 1.0,  0.0]),
    'Meling':       np.array([ 1.8, -0.5]),
    'Aursnes':      np.array([ 2.2,  0.3]),
    'Berge':        np.array([ 2.5,  0.0]),
    'Odegaard':     np.array([ 2.8,  0.5]),
    'Hauge':        np.array([ 2.0, -1.5]),
    'Haaland':      np.array([ 6.0,  1.0]),
    'Nusa':         np.array([ 2.5, -3.0]),
    'Maignan':      np.array([ 0.0,  0.0]),
    'Kounde':       np.array([-2.2, -0.8]),
    'Konate':       np.array([-1.0,  0.0]),
    'Saliba':       np.array([-1.0,  0.0]),
    'T-Hernandez':  np.array([-1.8,  0.5]),
    'Griezmann':    np.array([-2.8, -0.5]),
    'Tchouameni':   np.array([-2.5,  0.0]),
    'Kante':        np.array([-2.2,  0.0]),
    'Dembele':      np.array([-2.0,  1.5]),
    'Mbappe':       np.array([-6.0, -1.0]),
    'Olise':        np.array([-2.5,  3.0]),
}


# ── Voronoi areas ─────────────────────────────────────────────────────────────

# Precompute pitch grid (2 m resolution)
_gx, _gy = np.meshgrid(np.arange(1, 105, 2), np.arange(1, 68, 2))
_GRID     = np.column_stack([_gx.ravel(), _gy.ravel()])  # (N, 2)
_CELL     = 4.0   # m²

def voronoi_areas(snap_pos, team):
    """
    Approximate Voronoi area [m²] for each player on `team` at this snapshot.

    For each cell in a 2 m pitch grid, find the nearest teammate.
    Area = count of cells * cell_area.
    """
    names    = [p[0] for p in PLAYERS if p[1] == team]
    pts      = np.array([snap_pos[n] for n in names])      # (K, 2)
    dists    = np.linalg.norm(_GRID[:, None] - pts[None], axis=2)  # (N, K)
    nearest  = dists.argmin(axis=1)
    return {n: float((nearest == i).sum() * _CELL) for i, n in enumerate(names)}


# ── Pass success probability ───────────────────────────────────────────────────

def pass_prob(from_pos, to_pos, snap_pos, opp_team, lane_radius=1.5):
    """
    Probability that a pass from `from_pos` to `to_pos` succeeds.

    Decreases with distance and with the number of opponents within
    `lane_radius` metres of the passing lane.
    """
    vec  = to_pos - from_pos
    dist = np.linalg.norm(vec)

    # Count defenders in the lane
    n_def = 0
    if dist > 0:
        for name, team, *_ in PLAYERS:
            if team != opp_team:
                continue
            opp  = snap_pos[name]
            t    = np.clip(np.dot(opp - from_pos, vec) / dist**2, 0, 1)
            if np.linalg.norm(opp - (from_pos + t * vec)) < lane_radius:
                n_def += 1

    p_dist = np.exp(-dist / 60.0)   # halves every ~42 m
    p_def  = 0.85 ** n_def          # each defender cuts by 15%
    return float(max(0.75, p_dist * p_def))


# ── Main simulation ────────────────────────────────────────────────────────────

def simulate(n_steps=N_STEPS, dt=DT, p_event=0.009, seed=42):
    """
    Returns
    -------
    positions  : dict  name -> (n_steps, 2) true positions [m]
    possession : list  length n_steps, name of player with ball
    passes     : list  of dicts {step, from, to, success}
    """
    rng = np.random.default_rng(seed)

    # Initialise positions
    positions = {p[0]: np.zeros((n_steps, 2)) for p in PLAYERS}
    for name, team, x0, y0, *_ in PLAYERS:
        positions[name][0] = [x0, y0]

    # Random starting possessor from all outfield players
    outfield  = [p[0] for p in PLAYERS if p[0] not in ('Nyland', 'Maignan')]
    possessor = str(rng.choice(outfield))
    print(f'Starting poss : {possessor}  ({PLAYER_TEAM[possessor]})')
    possession = [possessor] * n_steps
    passes     = []

    for i in range(1, n_steps):
        snap = {name: positions[name][i-1] for name in positions}

        # ── Move all players ──────────────────────────────────────────
        poss_team = PLAYER_TEAM[possessor]
        ball_pos  = snap[possessor]

        for name, team, *_ in PLAYERS:
            if team == poss_team:
                vel = ATTACK_VEL[name].copy()   # advance toward opponent's goal
            else:
                # Press toward the ball
                to_ball = ball_pos - snap[name]
                dist    = np.linalg.norm(to_ball)
                vel     = (to_ball / dist * 4.0) if dist > 1 else np.zeros(2)

            noise   = rng.normal(0, PLAYER_SIG[name], 2)
            new_pos = snap[name] + (vel + noise) * dt
            positions[name][i] = np.clip(new_pos, [1, 1], [104, 67])

        # ── Possession event ──────────────────────────────────────────
        if rng.random() < p_event:
            snap_now  = {name: positions[name][i] for name in positions}
            poss_team = PLAYER_TEAM[possessor]
            opp_team  = 'FRA' if poss_team == 'NOR' else 'NOR'

            # Voronoi weights for teammates
            areas     = voronoi_areas(snap_now, poss_team)
            teammates = [n for n, t, *_ in PLAYERS
                         if t == poss_team and n != possessor]
            weights   = np.array([areas[n] for n in teammates])
            weights   = weights / weights.sum()

            # Sample target along this edge
            target = rng.choice(teammates, p=weights)

            # Pass success
            p_succ = pass_prob(
                snap_now[possessor], snap_now[target], snap_now, opp_team)

            if rng.random() < p_succ:
                passes.append(dict(step=i, frm=possessor,
                                   to=target, success=True))
                possessor = target
            else:
                # Error — nearest opponent intercepts
                opps   = [n for n, t, *_ in PLAYERS if t == opp_team]
                ball   = snap_now[target]
                dists  = [np.linalg.norm(snap_now[n] - ball) for n in opps]
                possessor = opps[int(np.argmin(dists))]
                passes.append(dict(step=i, frm=possessor,
                                   to=target, success=False))

        possession[i] = possessor

    return positions, possession, passes


def add_gps_noise(positions, sigma=0.8, seed=0):
    """Zero-mean Gaussian GPS noise, independent per player and axis."""
    rng = np.random.default_rng(seed)
    return {
        name: pos + rng.normal(0, sigma, pos.shape)
        for name, pos in positions.items()
    }


if __name__ == '__main__':
    positions, possession, passes = simulate(seed=None)
    noisy = add_gps_noise(positions)

    nor_passes  = sum(1 for p in passes if PLAYER_TEAM[p['frm']] == 'NOR')
    fra_passes  = sum(1 for p in passes if PLAYER_TEAM[p['frm']] == 'FRA')
    nor_success = sum(1 for p in passes if PLAYER_TEAM[p['frm']] == 'NOR' and p['success'])
    fra_success = sum(1 for p in passes if PLAYER_TEAM[p['frm']] == 'FRA' and p['success'])
    success     = nor_success + fra_success

    print(f'Steps      : {N_STEPS}  ({N_STEPS * DT:.1f} s)')
    print(f'Passes     : {len(passes)}  (NOR {nor_passes}  FRA {fra_passes})')
    print(f'Success    : {success}/{len(passes)}  ({100*success/max(len(passes),1):.0f}%)')
    print(f'  NOR      : {nor_success}/{nor_passes}  ({100*nor_success/max(nor_passes,1):.0f}%)')
    print(f'  FRA      : {fra_success}/{fra_passes}  ({100*fra_success/max(fra_passes,1):.0f}%)')
    print(f'Final poss : {possession[-1]}  ({PLAYER_TEAM[possession[-1]]})')
