import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_stats(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_reward_plot(stats, run_tag: str, artifact_dir: Path):
    if not stats:
        return

    episodes = [item["episode"] for item in stats]
    rewards = [item["reward_total"] for item in stats]

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, marker="o", linewidth=1.5)
    plt.title(f"SAC Reward Curve ({run_tag})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifact_dir / f"reward_curve_{run_tag}.png", dpi=140)
    plt.close()


def save_lap_plot(stats, run_tag: str, artifact_dir: Path):
    lap_rows = [item for item in stats if item.get("lap_time_sec") is not None]
    if not lap_rows:
        return

    episodes = [item["episode"] for item in lap_rows]
    lap_times = [item["lap_time_sec"] for item in lap_rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, lap_times, marker="o", linewidth=1.5, color="tab:green")
    plt.title(f"SAC Lap Time Trend ({run_tag})")
    plt.xlabel("Episode")
    plt.ylabel("Lap Time (s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(artifact_dir / f"lap_time_{run_tag}.png", dpi=140)
    plt.close()


def save_summary(train_stats, eval_stats, artifact_dir: Path):
    def summarize(stats):
        if not stats:
            return {
                "episodes": 0,
                "completions": 0,
                "completion_rate": 0.0,
                "best_reward": None,
                "best_lap_time_sec": None,
            }

        completed = [item for item in stats if item.get("finished")]
        lap_times = [item["lap_time_sec"] for item in completed if item.get("lap_time_sec") is not None]
        rewards = [item["reward_total"] for item in stats]
        return {
            "episodes": len(stats),
            "completions": len(completed),
            "completion_rate": len(completed) / len(stats),
            "best_reward": max(rewards) if rewards else None,
            "best_lap_time_sec": min(lap_times) if lap_times else None,
        }

    summary = {
        "train": summarize(train_stats),
        "eval": summarize(eval_stats),
    }
    with (artifact_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_stats = load_stats(artifact_dir / "episode_stats_train.json")
    eval_stats = load_stats(artifact_dir / "episode_stats_eval.json")

    save_reward_plot(train_stats, "train", artifact_dir)
    save_reward_plot(eval_stats, "eval", artifact_dir)
    save_lap_plot(train_stats, "train", artifact_dir)
    save_lap_plot(eval_stats, "eval", artifact_dir)
    save_summary(train_stats, eval_stats, artifact_dir)


if __name__ == "__main__":
    main()
