# Norway vs France — FIFA World Cup 2026
## Real-Time Player & Ball Tracking Pipeline

A Kalman filter data processing pipeline demonstrated on simulated tracking
data from today's Group I decider (June 26, 2026). Two players and three
scenarios: Haaland's striker run, Olise's free kick, and simultaneous
multi-target tracking with full player dynamics analysis.

---

## Processing Levels

| Level | Name | Description |
|-------|------|-------------|
| **L0** | Raw measurements | Noisy GPS (player) and camera (ball) position observations |
| **L1** | Kalman filter | Recursive state estimation — filtered position + latent velocity |
| **L2** | Prediction | N-step-ahead trajectory; arrival/landing point before it happens |

---

## Scenarios

### [1] Haaland Striker Run  *(1st half — Norway attacking)*

Erling Haaland makes a diagonal run from the centre circle into the French
penalty area. A GPS vest worn by each player records noisy position data
(σ ≈ 1.2 m). The Kalman filter recovers the clean trajectory and estimates
speed and heading in real time. After 5.5 seconds of tracking, it predicts
Haaland's position **2 seconds ahead** — telling a teammate where to pass
before he arrives.

**Key result:** Estimated speed 31.7 km/h. 2-second prediction error < 1 m.

### [2] Olise Free Kick  *(2nd half — France attacking)*

Michael Olise strikes a free kick from 25 m out. The broadcast camera system
observes the ball position at 20 Hz with noise σ ≈ 0.18 m. After seeing only
**15 frames (0.75 seconds)** of flight — before the ball is halfway to goal —
the filter predicts the complete trajectory and determines:

> **GOAL** — ball arrives at (105.4 m, 34.2 m, 0.59 m), center of goal.

This early-prediction capability mirrors how real tracking systems issue
alerts before a target arrives at a location of interest.

### [3 & 4] Multi-Target Tracking + Player Dynamics

All three players (Haaland, Olise, Konate) tracked simultaneously from GPS
data. The KF velocity state — never directly observed — is used to derive:

- Speed [km/h] over time for each player
- Heading angle [degrees]
- Acceleration magnitude [m/s²]

This demonstrates the filter's core strength: recovering latent kinematic
states (velocity, acceleration) from position-only measurements.

---

## Algorithm

### Motion Model

State vector `x = [px, py, vx, vy]ᵀ` (2-D) or `x = [px, py, pz, vx, vy, vz]ᵀ` (3-D).

```
Predict:  x_{k|k-1} = F x_{k-1|k-1} + u          u = gravity input (ball only)
          P_{k|k-1} = F P_{k-1|k-1} Fᵀ + Q

Update:   innovation = z_k − H x_{k|k-1}
          K          = P_{k|k-1} Hᵀ (H P_{k|k-1} Hᵀ + R)⁻¹
          x_{k|k}    = x_{k|k-1} + K · innovation
          P_{k|k}    = (I − KH) P_{k|k-1}
```

### N-Step-Ahead Prediction

```
x_{k+j|k} = Fʲ x_{k|k} + Σᵢ₌₀ʲ⁻¹ Fⁱ u      (j = prediction horizon)
```

The prediction is propagated from the current filtered state using the same
motion model. Uncertainty grows with horizon length (see Figure 3).

---

## Assumptions

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | Constant-velocity dynamics | Standard first-order kinematic model; manoeuvres enter through process noise Q |
| 2 | Zero-mean Gaussian measurement noise | Central limit theorem applies to GPS and optical sensor noise |
| 3 | Position-only observations | Velocity sensors not standard in match tracking systems |
| 4 | Independent noise in x and y (diagonal R) | Sensor axes are physically independent |
| 5 | Gravity known; drag unmodelled (ball) | Gravity is deterministic; drag depends on spin/turbulence and is absorbed into Q |
| 6 | Sigmoid acceleration profile (players) | Models standing-start sprint; intentional model mismatch with constant-velocity KF (realistic) |
| 7 | All data is simulated | No real player tracking data used; kinematics calibrated to published performance figures |

**Player kinematics calibration:**
- Haaland top speed: ~35 km/h (published GPS vest data, simulated at 31.7 km/h peak)
- Olise sprint speed: ~36 km/h (Bundesliga 2025-26 tracking data)
- Ball initial speed: ~94 km/h (typical professional free kick)

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
| `--sigma-gps` | 0.5 | Player GPS noise std dev [m] |
| `--sigma-cam` | 0.18 | Ball camera noise std dev [m] |
| `--pred-at` | 55 | Step at which to issue Haaland prediction |
| `--obs-fk` | 15 | Camera frames before free kick prediction |
| `--seed` | 42 | Random seed |
| `--out` | outputs | Output directory |

**Example — noisier GPS:**
```bash
python wc_tracker.py --sigma-gps 2.0
```

---

## Outputs

| File | Description |
|------|-------------|
| `01_haaland_run.png` | Haaland's run on pitch: GPS scatter, KF track, 2-s prediction, covariance ellipses |
| `02_olise_freekick.png` | Free kick side view + top view; predicted trajectory from 15 frames |
| `03_prediction_horizon.png` | Prediction error vs horizon length — uncertainty grows with time |
| `04_multi_player.png` | All three players tracked simultaneously on pitch |
| `05_player_dynamics.png` | Speed, heading, and acceleration derived from KF velocity state |
| `metrics.json` | All numerical results |

---

## Results (default seed=42, σ_GPS=1.2 m)

```
Haaland run
  Raw GPS RMSE       : 1.025 m
  KF RMSE            : 0.906 m   (11.6% improvement)
  Estimated speed    : 31.7 km/h
  2-s prediction err : 0.939 m

Olise free kick
  Camera noise RMSE  : 0.142 m
  KF RMSE            : 0.138 m
  Observation window : 0.75 s (15 frames)
  Prediction         : GOAL
```

---

## References

- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley. (Process noise model: Ch. 6)
- Li, X. R., & Jilkov, V. P. (2003). Survey of maneuvering target tracking. *IEEE Trans. Aerospace and Electronic Systems*, 39(4). (Sinusoidal benchmark trajectory)
- FIFA (2018). *Laws of the Game*. (Pitch dimensions: 105 m × 68 m, goal 7.32 m × 2.44 m)
