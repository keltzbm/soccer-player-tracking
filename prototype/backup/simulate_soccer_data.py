import numpy as np
import pandas as pd
from pathlib import Path


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def formation_positions(team: str):
    """Simple 4-3-3 style formation."""
    if team == "home":
        x_base = [10, 25, 25, 25, 25, 45, 45, 45, 65, 65, 75]
    else:
        x_base = [95, 80, 80, 80, 80, 60, 60, 60, 40, 40, 30]

    y_base = [34, 10, 25, 43, 58, 18, 34, 50, 20, 48, 34]

    return np.array(list(zip(x_base, y_base)), dtype=float)


def simulate_match(
    n_frames=3000,
    fps=10,
    noise_std=0.75,
    missing_prob=0.02,
    seed=42,
):
    rng = np.random.default_rng(seed)

    rows_truth = []
    rows_observed = []

    home_base = formation_positions("home")
    away_base = formation_positions("away")

    ball = np.array([52.5, 34.0])
    ball_velocity = np.array([0.08, 0.02])

    for frame in range(n_frames):
        time = frame / fps

        # Ball movement with random perturbations
        if frame % 200 == 0:
            ball_velocity = rng.normal(0, 0.25, size=2)

        ball += ball_velocity
        ball[0] = np.clip(ball[0], 0, FIELD_LENGTH)
        ball[1] = np.clip(ball[1], 0, FIELD_WIDTH)

        possession_team = "home" if ball[0] < FIELD_LENGTH / 2 else "away"

        for team, base_positions in [("home", home_base), ("away", away_base)]:
            attacking_direction = 1 if team == "home" else -1

            for player_idx, base in enumerate(base_positions):
                player_id = f"{team}_{player_idx + 1:02d}"

                # Smooth oscillatory motion
                phase = 0.4 * player_idx
                dx = 3.0 * np.sin(0.015 * frame + phase)
                dy = 2.0 * np.cos(0.012 * frame + phase)

                # Team shifts toward ball
                ball_influence = 0.08 * (ball - base)

                # Attacking team pushes forward slightly
                attack_push = np.array([4.0 * attacking_direction, 0.0])
                if possession_team != team:
                    attack_push *= -0.4

                true_pos = base + np.array([dx, dy]) + ball_influence + attack_push

                true_pos[0] = np.clip(true_pos[0], 0, FIELD_LENGTH)
                true_pos[1] = np.clip(true_pos[1], 0, FIELD_WIDTH)

                rows_truth.append({
                    "frame": frame,
                    "time": time,
                    "team": team,
                    "player_id": player_id,
                    "true_x": true_pos[0],
                    "true_y": true_pos[1],
                })

                # Observed tracking data: noisy and occasionally missing
                if rng.random() > missing_prob:
                    obs = true_pos + rng.normal(0, noise_std, size=2)

                    rows_observed.append({
                        "frame": frame,
                        "time": time,
                        "team": team,
                        "player_id": player_id,
                        "obs_x": np.clip(obs[0], 0, FIELD_LENGTH),
                        "obs_y": np.clip(obs[1], 0, FIELD_WIDTH),
                    })

        rows_truth.append({
            "frame": frame,
            "time": time,
            "team": "ball",
            "player_id": "ball",
            "true_x": ball[0],
            "true_y": ball[1],
        })

        rows_observed.append({
            "frame": frame,
            "time": time,
            "team": "ball",
            "player_id": "ball",
            "obs_x": ball[0] + rng.normal(0, noise_std),
            "obs_y": ball[1] + rng.normal(0, noise_std),
        })

    return pd.DataFrame(rows_truth), pd.DataFrame(rows_observed)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    truth_df, observed_df = simulate_match()

    truth_df.to_csv("data/truth_tracks.csv", index=False)
    observed_df.to_csv("data/raw_tracking_data.csv", index=False)

    print("Synthetic soccer tracking data generated.")
    print(f"Truth rows: {len(truth_df)}")
    print(f"Observed rows: {len(observed_df)}")
