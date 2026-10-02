"""
filter.py
---------
Kalman filter for 11-player GPS tracking data.

One independent filter per player. Each filter takes noisy GPS
position observations (L0) and produces:

  - Filtered position estimates (L1)
  - Latent velocity estimates (L1) — never directly observed
  - Per-player state covariance (uncertainty)

The team centroid is then computed from the filtered positions,
producing a smoother collective signal than the raw GPS centroid.

Algorithm
---------
State   x = [px, py, vx, vy]^T

Predict   x_{k|k-1} = F x_{k-1|k-1}
          P_{k|k-1} = F P_{k-1|k-1} F^T + Q

Update    y = z_k - H x_{k|k-1}               (innovation)
          S = H P_{k|k-1} H^T + R              (innovation covariance)
          K = P_{k|k-1} H^T S^{-1}             (Kalman gain)
          x_{k|k} = x_{k|k-1} + K y
          P_{k|k} = (I - KH) P_{k|k-1}

Assumptions
-----------
- Constant-velocity dynamics (linear, time-invariant)
- Position-only observations; velocity is a latent state
- Zero-mean white Gaussian process and measurement noise (DWNA model)
- Noise is independent across players
- GPS noise sigma = 0.8 m (sensor inaccuracy, separate from game noise)
"""

import numpy as np
from simulate import simulate, add_gps_noise, PLAYERS, DT


class KalmanFilter:
    """
    2-D constant-velocity Kalman filter.

    Parameters
    ----------
    dt      : float  Time step [s]
    sigma_q : float  Process noise intensity [m/s^2]
              Models unobserved accelerations — the game noise that
              the constant-velocity model cannot predict.
    sigma_r : float  Measurement noise std dev [m]
              GPS sensor inaccuracy.
    """

    def __init__(self, dt, sigma_q, sigma_r):
        # State transition (constant velocity)
        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]], dtype=float)

        # Observation (position only)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Process noise — DWNA model (Bar-Shalom et al., 2001)
        q = sigma_q ** 2
        self.Q = q * np.array([
            [dt**4/4,       0, dt**3/2,       0],
            [      0, dt**4/4,       0, dt**3/2],
            [dt**3/2,       0,   dt**2,       0],
            [      0, dt**3/2,       0,   dt**2]])

        # Measurement noise
        self.R = (sigma_r ** 2) * np.eye(2)

    def run(self, measurements):
        """
        Forward Kalman filter pass.

        Parameters
        ----------
        measurements : (N, 2)  noisy GPS position observations

        Returns
        -------
        xs : (N, 4)    filtered state estimates [px, py, vx, vy]
        Ps : (N, 4, 4) state covariances
        """
        N  = len(measurements)
        xs = np.zeros((N, 4))
        Ps = np.zeros((N, 4, 4))

        # Initialise from first measurement, zero velocity
        x = np.array([measurements[0, 0], measurements[0, 1], 0., 0.])
        P = np.diag([1., 1., 25., 25.])

        for k, z in enumerate(measurements):
            # Predict
            x_p = self.F @ x
            P_p = self.F @ P @ self.F.T + self.Q

            # Update
            y   = z - self.H @ x_p
            S   = self.H @ P_p @ self.H.T + self.R
            K   = P_p @ self.H.T @ np.linalg.inv(S)
            x   = x_p + K @ y
            P   = (np.eye(4) - K @ self.H) @ P_p

            xs[k], Ps[k] = x, P

        return xs, Ps


def run_all(noisy, dt=DT, sigma_q=3.0, sigma_r=0.8):
    """
    Run one Kalman filter per player.

    Parameters
    ----------
    noisy   : dict  name -> (N, 2) noisy GPS measurements
    sigma_q : float Process noise — tune to match role-based game noise
    sigma_r : float GPS sensor noise

    Returns
    -------
    filtered : dict  name -> {'xs': (N,4), 'Ps': (N,4,4)}
    """
    kf       = KalmanFilter(dt, sigma_q, sigma_r)
    filtered = {}

    for name in noisy:
        xs, Ps          = kf.run(noisy[name])
        filtered[name]  = {'xs': xs, 'Ps': Ps}

    return filtered


def filtered_centroid(filtered):
    """
    Compute team centroid from filtered position estimates.

    Returns
    -------
    centroid : (N, 2)  smoothed team centroid [m]
    """
    positions = np.stack([filtered[p[0]]['xs'][:, :2]
                          for p in PLAYERS], axis=0)   # (11, N, 2)
    return positions.mean(axis=0)                       # (N, 2)


def raw_centroid(noisy):
    """Team centroid from raw noisy GPS measurements."""
    positions = np.stack([noisy[p[0]] for p in PLAYERS], axis=0)
    return positions.mean(axis=0)


if __name__ == '__main__':
    positions, centroid, t = simulate()
    noisy    = add_gps_noise(positions)
    filtered = run_all(noisy)
    cent_raw  = raw_centroid(noisy)
    cent_filt = filtered_centroid(filtered)

    # Per-player RMSE
    print(f'{"Player":<6}  {"Role":<4}  {"Raw RMSE":>10}  {"KF RMSE":>10}  {"Improvement":>12}')
    print('-' * 52)
    for name, role, *_ in PLAYERS:
        true  = positions[name]
        raw   = noisy[name]
        filt  = filtered[name]['xs'][:, :2]
        rmse_raw  = float(np.sqrt(((raw  - true)**2).mean()))
        rmse_filt = float(np.sqrt(((filt - true)**2).mean()))
        impr = 100 * (1 - rmse_filt / rmse_raw)
        print(f'{name:<6}  {role:<4}  {rmse_raw:>10.3f}  '
              f'{rmse_filt:>10.3f}  {impr:>11.1f}%')

    # Centroid RMSE
    print()
    rmse_raw_c  = float(np.sqrt(((cent_raw  - centroid)**2).mean()))
    rmse_filt_c = float(np.sqrt(((cent_filt - centroid)**2).mean()))
    impr_c = 100 * (1 - rmse_filt_c / rmse_raw_c)
    print(f'{"Centroid":<6}        {rmse_raw_c:>10.3f}  '
          f'{rmse_filt_c:>10.3f}  {impr_c:>11.1f}%')
