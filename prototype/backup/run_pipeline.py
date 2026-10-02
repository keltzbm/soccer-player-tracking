# run_pipeline.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def moving_average(x, window=9):
    return x.rolling(window=window, center=True, min_periods=1).mean()


def process_tracks(raw_df, window=9):
    processed = []

    for player_id, g in raw_df.groupby("player_id"):
        g = g.sort_values("time").copy()

        g["x_smooth"] = moving_average(g["obs_x"], window)
        g["y_smooth"] = moving_average(g["obs_y"], window)

        g["vx"] = np.gradient(g["x_smooth"], g["time"])
        g["vy"] = np.gradient(g["y_smooth"], g["time"])
        g["speed"] = np.sqrt(g["vx"] ** 2 + g["vy"] ** 2)

        g["ax"] = np.gradient(g["vx"], g["time"])
        g["ay"] = np.gradient(g["vy"], g["time"])
        g["acceleration"] = np.sqrt(g["ax"] ** 2 + g["ay"] ** 2)

        processed.append(g)

    return pd.concat(processed, ignore_index=True)


def compute_team_metrics(processed_df):
    rows = []

    for (frame, team), g in processed_df[processed_df["team"].isin(["home", "away"])].groupby(["frame", "team"]):
        cx = g["x_smooth"].mean()
        cy = g["y_smooth"].mean()

        width = g["y_smooth"].max() - g["y_smooth"].min()
        depth = g["x_smooth"].max() - g["x_smooth"].min()

        compactness = np.mean(np.sqrt((g["x_smooth"] - cx) ** 2 + (g["y_smooth"] - cy) ** 2))

        rows.append({
            "frame": frame,
            "time": g["time"].iloc[0],
            "team": team,
            "centroid_x": cx,
            "centroid_y": cy,
            "team_width": width,
            "team_depth": depth,
            "compactness": compactness,
            "mean_speed": g["speed"].mean(),
        })

    return pd.DataFrame(rows)


def evaluate_against_truth(processed_df, truth_df):
    truth = truth_df.rename(columns={"true_x": "truth_x", "true_y": "truth_y"})

    merged = processed_df.merge(
        truth[["frame", "player_id", "truth_x", "truth_y"]],
        on=["frame", "player_id"],
        how="inner",
    )

    merged["raw_error"] = np.sqrt(
        (merged["obs_x"] - merged["truth_x"]) ** 2 +
        (merged["obs_y"] - merged["truth_y"]) ** 2
    )

    merged["smooth_error"] = np.sqrt(
        (merged["x_smooth"] - merged["truth_x"]) ** 2 +
        (merged["y_smooth"] - merged["truth_y"]) ** 2
    )

    metrics = {
        "raw_position_rmse_m": float(np.sqrt(np.mean(merged["raw_error"] ** 2))),
        "smoothed_position_rmse_m": float(np.sqrt(np.mean(merged["smooth_error"] ** 2))),
        "mean_speed_m_per_s": float(processed_df[processed_df["team"] != "ball"]["speed"].mean()),
        "max_speed_m_per_s": float(processed_df[processed_df["team"] != "ball"]["speed"].max()),
    }

    return merged, metrics


def draw_pitch():
    plt.xlim(0, FIELD_LENGTH)
    plt.ylim(0, FIELD_WIDTH)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)


def plot_raw_vs_smooth(processed_df, player_id):
    g = processed_df[processed_df["player_id"] == player_id].sort_values("time")

    plt.figure(figsize=(10, 6))
    draw_pitch()
    plt.scatter(g["obs_x"], g["obs_y"], s=8, alpha=0.4, label="Raw observations")
    plt.plot(g["x_smooth"], g["y_smooth"], linewidth=2, label="Smoothed track")
    plt.title(f"Raw vs smoothed trajectory: {player_id}")
    plt.legend()
    plt.savefig(f"outputs/raw_vs_smooth_{player_id}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_speed(processed_df, player_id):
    g = processed_df[processed_df["player_id"] == player_id].sort_values("time")

    plt.figure(figsize=(10, 4))
    plt.plot(g["time"], g["speed"])
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.title(f"Estimated speed over time: {player_id}")
    plt.grid(alpha=0.25)
    plt.savefig(f"outputs/speed_{player_id}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_team_metrics(team_metrics):
    plt.figure(figsize=(10, 4))

    for team, g in team_metrics.groupby("team"):
        plt.plot(g["time"], g["compactness"], label=team)

    plt.xlabel("time (s)")
    plt.ylabel("compactness (m)")
    plt.title("Team compactness over time")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig("outputs/team_compactness.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_histogram(eval_df):
    plt.figure(figsize=(8, 5))
    plt.hist(eval_df["raw_error"], bins=40, alpha=0.5, label="Raw")
    plt.hist(eval_df["smooth_error"], bins=40, alpha=0.5, label="Smoothed")
    plt.xlabel("position error (m)")
    plt.ylabel("count")
    plt.title("Position error before and after smoothing")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.savefig("outputs/error_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    Path("outputs").mkdir(exist_ok=True)

    raw_df = pd.read_csv("data/raw_tracking_data.csv")
    truth_df = pd.read_csv("data/truth_tracks.csv")

    processed_df = process_tracks(raw_df, window=9)
    team_metrics = compute_team_metrics(processed_df)
    eval_df, metrics = evaluate_against_truth(processed_df, truth_df)

    processed_df.to_csv("data/processed_tracks.csv", index=False)
    team_metrics.to_csv("data/team_metrics.csv", index=False)
    eval_df.to_csv("data/evaluation_errors.csv", index=False)

    pd.Series(metrics).to_json("outputs/metrics.json", indent=2)

    plot_raw_vs_smooth(processed_df, "home_09")
    plot_speed(processed_df, "home_09")
    plot_team_metrics(team_metrics)
    plot_error_histogram(eval_df)

    print("Pipeline complete.")
    print(metrics)


if __name__ == "__main__":
    main()
