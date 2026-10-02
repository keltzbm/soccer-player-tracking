import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIELD_LENGTH = 105
FIELD_WIDTH = 68


def draw_pitch():
    plt.xlim(0, FIELD_LENGTH)
    plt.ylim(0, FIELD_WIDTH)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)


def plot_frame(df, frame):
    frame_df = df[df["frame"] == frame]

    plt.figure(figsize=(10, 6))
    draw_pitch()

    for team, color in [("home", "blue"), ("away", "red"), ("ball", "black")]:
        team_df = frame_df[frame_df["team"] == team]
        plt.scatter(team_df["obs_x"], team_df["obs_y"], label=team, s=50)

        for _, row in team_df.iterrows():
            if row["team"] != "ball":
                plt.text(row["obs_x"] + 0.5, row["obs_y"] + 0.5, row["player_id"], fontsize=7)

    plt.title(f"Observed player positions at frame {frame}")
    plt.legend()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(f"outputs/frame_{frame}.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_player_trajectory(df, player_id):
    player_df = df[df["player_id"] == player_id]

    plt.figure(figsize=(10, 6))
    draw_pitch()
    plt.plot(player_df["obs_x"], player_df["obs_y"], marker=".", linewidth=1)
    plt.title(f"Observed trajectory for {player_id}")
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(f"outputs/trajectory_{player_id}.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_all_trajectories(df):
    plt.figure(figsize=(10, 6))
    draw_pitch()

    for player_id, player_df in df[df["team"] != "ball"].groupby("player_id"):
        plt.plot(player_df["obs_x"], player_df["obs_y"], linewidth=0.8, alpha=0.7)

    ball_df = df[df["player_id"] == "ball"]
    plt.plot(ball_df["obs_x"], ball_df["obs_y"], linewidth=2, label="ball")

    plt.title("Observed trajectories for all players")
    plt.legend()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/all_observed_trajectories.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/raw_tracking_data.csv")

    plot_frame(df, frame=100)
    plot_player_trajectory(df, player_id="home_09")
    plot_all_trajectories(df)
