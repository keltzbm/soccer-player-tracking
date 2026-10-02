# Norway vs France — FIFA World Cup 2026, Group I
## 22-Player Kalman Filter Tracking Pipeline

A data processing pipeline that models a professional football match as a
system of 22 simultaneously tracked targets. One independent Kalman filter
per player; team-level analytics derived from the ensemble of filtered
state estimates. Framed around today's Group I decider (June 26, 2026).

---

## Processing Levels

| Level | Name | Description |
|-------|------|-------------|
| **L0** | Raw measurements | Noisy GPS position observations from player vests (σ ≈ 0.8 m) and broadcast camera (σ ≈ 0.18 m) |
| **L1** | Kalman filter | Per-player recursive state estimation — filtered position + latent velocity |
| **L2** | Analytics | Team centroid, compactness, press intensity; speed distribution; free kick trajectory prediction |

---

## Scenarios

### [1] 22-Player System — Norwegian Attacking Phase

A 20-second phase of play with Norway pressing into the French half. Each
of the 22 players is assigned a role-based trajectory and tracked
independently by its own Kalman filter.

The filter state vector `x = [px, py, vx, vy]ᵀ` recovers both position
and velocity from position-only GPS observations. Velocity is a **latent
state** — never directly measured, fully inferred by the filter.

From the ensemble of 22 filtered tracks, three team-level quantities
are derived (L2):

- **Team centroid x** — how advanced each team is on the pitch
- **Team compactness** — mean spread; lower = more compact / organised
- **Press intensity** — rate at which France closes space toward Norway

**Key result:** Mean KF RMSE 0.362 m vs raw GPS RMSE ~1.131 m — **68%
improvement** across all 22 players simultaneously.

### [2] Olise Free Kick — 3-D Ball Tracking

Michael Olise strikes a free kick from 25 m out. A broadcast camera
observes the ball position at 20 Hz (σ ≈ 0.18 m). After seeing only
**15 frames (0.75 seconds)** of flight, the filter predicts the
complete trajectory and issues a verdict:

> **GOAL** — ball arrives at (105.4 m, 34.2 m, 0.59 m).

This mirrors how real tracking systems issue early-warning predictions
before a target arrives at a location of interest.

---

## Algorithm

### Motion Model (Player Tracking — 2-D)

State `x = [px, py, vx, vy]ᵀ`

```
Predict   x_{k|k-1} = F x_{k-1|k-1}
          P_{k|k-1} = F P_{k-1|k-1} F^T + Q

Update    innovation = z_k - H x_{k|k-1}
          K          = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}
          x_{k|k}    = x_{k|k-1} + K * innovation
          P_{k|k}    = (I - KH) P_{k|k-1}
```

State transition `F` implements constant-velocity kinematics. Process
noise `Q` uses the discretised white-noise acceleration (DWNA) model,
which absorbs unmodelled player accelerations and direction changes.

### Motion Model (Ball Tracking — 3-D)

State `x = [px, py, pz, vx, vy, vz]ᵀ`. Gravity is applied as a known
control input `u` at each predict step:

```
x_{k|k-1} = F x_{k-1|k-1} + u       u = [0, 0, -½g dt², 0, 0, -g dt]^T
```

Aerodynamic drag (nonlinear in ball speed) is intentionally unmodelled
in the filter — it enters as model mismatch and is absorbed into Q. This
reflects real operational conditions where drag coefficients are uncertain.

### N-Step-Ahead Prediction

```
x_{k+j|k} = F^j x_{k|k} + sum_{i=0}^{j-1} F^i u      (j = horizon)
```

Applied to the free kick: after 15 frames the filter propagates the
state estimate forward until the ball crosses the goal line.

---

## Player Simulation

Each player's trajectory is generated using **Langevin dynamics**:

```
v_{k+1} = alpha * v_k + (1 - alpha) * v_target(t) + N(0, sigma_v)
x_{k+1} = x_{k} + v_{k+1} * dt
```

The smoothing parameter `alpha = 0.82` produces correlated, realistic
movement. `v_target` is role-dependent:

| Role | Behaviour |
|------|-----------|
| GK | Near-stationary; tiny random jitter |
| CB/FB | Slow drift; hold defensive shape |
| CM | Moderate advance tracking ball area |
| Haaland (ST) | Sigmoid sprint profile from standstill |
| Nusa (RW) | Straight run → sharp inside cut at t = 7 s |
| Konate (CB) | Dynamically tracks Haaland with 0.8 s reaction delay |
| Mbappe (ST) | High-intensity press toward Norway |

Model mismatch between the Langevin dynamics and the constant-velocity
KF is **intentional** — it reflects the reality that no motion model
perfectly captures player behaviour.

---

## Assumptions

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | Constant-velocity KF dynamics | Standard first-order kinematic model; manoeuvres handled by process noise Q |
| 2 | Gaussian GPS noise (σ = 0.8 m) | Consistent with published accuracy specs for wearable GPS vests in sport |
| 3 | Independent noise per player | Sensor hardware is physically separated |
| 4 | Position-only observations | Velocity not directly provided by standard GPS at 10 Hz |
| 5 | Gravity known; drag unmodelled (ball) | Gravity is deterministic; drag depends on spin/Reynolds number and enters Q |
| 6 | All trajectories simulated | No real player tracking data used; kinematics calibrated to published figures |

**Kinematic calibration sources:**
- Haaland top speed ~35 km/h (GPS vest data, professional match)
- Mbappe top speed ~36 km/h (Ligue 1 tracking data)
- GPS vest accuracy 0.3–1.5 m RMS depending on system (Catapult, STATSports)
- FIFA size-5 ball: m = 0.43 kg, r = 0.1085 m, C_D ≈ 0.20 (turbulent)

---

## Installation

```bash
pip install numpy matplotlib
```

## Usage

```bash
python wc_tracker.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma-gps` | 0.8 | Player GPS noise std dev [m] |
| `--sigma-cam` | 0.18 | Ball camera noise std dev [m] |
| `--snap-step` | 100 | Figure 1 snapshot time step (100 = 10 s) |
| `--obs-fk` | 15 | Camera frames before free kick prediction |
| `--seed` | 42 | Random seed |
| `--out` | outputs | Output directory |

**Example — noisier GPS:**
```bash
python wc_tracker.py --sigma-gps 1.5 --snap-step 80
```

---

## Outputs

| File | Description |
|------|-------------|
| `01_match_snapshot.png` | All 22 players on pitch at t = 10 s: GPS cloud, KF positions, velocity arrows, covariance ellipses |
| `02_team_dynamics.png` | Team centroid, compactness, and press intensity over 20 seconds |
| `03_player_dynamics.png` | Speed and heading for Haaland, Mbappe, Konate, Odegaard, Olise, Nusa |
| `04_speed_distribution.png` | All 22 player speeds at t = 10 s from KF velocity state |
| `05_olise_freekick.png` | 3-D free kick: side view + top view; predicted trajectory from 15 frames |
| `metrics.json` | All numerical results |

---

## Results (seed = 42, σ_GPS = 0.8 m)

```
22-Player System
  Raw GPS RMSE (approx) : 1.131 m
  Mean KF RMSE          : 0.362 m   (68.0% improvement)
  Haaland KF RMSE       : 0.527 m
  Konate  KF RMSE       : 0.702 m   (higher — dynamic tracking behaviour)

Olise Free Kick
  Camera noise RMSE     : 0.142 m
  Observation window    : 0.75 s  (15 frames at 20 Hz)
  Prediction            : GOAL
```

---

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley. (DWNA process noise model: Ch. 6)
- Li, X. R., & Jilkov, V. P. (2003). Survey of maneuvering target tracking. *IEEE Trans. Aerospace and Electronic Systems*, 39(4). (Model mismatch and benchmark trajectories)
- Catapult Sports (2022). *GPS and GNSS accuracy in sport: technical white paper.* (GPS vest accuracy: 0.3–1.5 m RMS)
- FIFA (2018). *Laws of the Game.* (Pitch: 105 m × 68 m; goal: 7.32 m × 2.44 m; ball specifications)
