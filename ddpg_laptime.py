"""
ddpg_laptime.py
────────────────
DDPG agent for TORCS whose objective is *fastest lap time*.

Key design decisions (from research):
  • Ben Lau reward: Vx*cos(angle) - |Vx*sin(angle)| - Vx*|trackPos|
  • Stochastic braking: brake explored only 10% of the time (Ben Lau's trick)
  • OU noise biased toward acceleration to prevent stuck-at-start local minima
  • No simultaneous accel+brake clamping during training — let the network learn

Usage
─────
  python ddpg_laptime.py --train 1 --episodes 500 --artifact-dir ./artifacts
  python ddpg_laptime.py --train 0 --episodes 10  --artifact-dir ./artifacts
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from gym_torcs_laptime import TorcsEnvLapTime
import numpy as np
import argparse
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

OU = OU()


def playGame(
    train_indicator=1,
    episode_count=500,
    max_steps=100000,
    artifact_dir=".",
    run_tag="laptime",
):
    artifact_dir = os.path.abspath(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    BUFFER_SIZE = 100_000
    BATCH_SIZE  = 32
    GAMMA       = 0.99
    TAU         = 0.001
    LRA         = 0.0001
    LRC         = 0.001

    action_dim = 3
    state_dim  = TorcsEnvLapTime.STATE_DIM   # 31

    np.random.seed(1337)

    EXPLORE      = 100_000.
    epsilon      = 1.0
    EPSILON_MIN  = 0.05
    step         = 0

    tf.compat.v1.disable_eager_execution()
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    config = tf.compat.v1.ConfigProto()
    config.device_count["GPU"] = 0
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    actor  = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff   = ReplayBuffer(BUFFER_SIZE)

    env = TorcsEnvLapTime(vision=False, throttle=True, gear_change=False)

    actor_weights  = os.path.join(artifact_dir, "actor_laptime.h5")
    critic_weights = os.path.join(artifact_dir, "critic_laptime.h5")
    print("Loading weights …")
    try:
        actor.model.load_weights(actor_weights)
        critic.model.load_weights(critic_weights)
        actor.target_model.load_weights(actor_weights)
        critic.target_model.load_weights(critic_weights)
        print("Weights loaded successfully.")
    except Exception:
        print("No saved weights found – starting from scratch.")

    episode_stats   = []
    all_lap_times   = []
    best_lap_time   = float('inf')

    print("TORCS Lap-Time Experiment start  (train=%d, episodes=%d)"
          % (train_indicator, episode_count))

    for i in range(episode_count):
        print("\nEpisode %d / %d  |  Replay buffer: %d  |  Best lap: %s s"
              % (i + 1, episode_count, buff.count(),
                 ("%.3f" % best_lap_time) if best_lap_time < float('inf') else "–"))

        if i % 3 == 0:
            s_t = env.reset(relaunch=True)
        else:
            s_t = env.reset()

        total_reward = 0.0
        episode_laps = []

        for j in range(max_steps):
            loss = 0.0

            epsilon = max(EPSILON_MIN, epsilon - 1.0 / EXPLORE)
            a_t        = np.zeros([1, action_dim])
            noise_t    = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(
                s_t.reshape(1, s_t.shape[0]), verbose=0
            )

            # Steering noise: moderate OU for exploration
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(
                a_t_original[0][0],  0.0,  0.60, 0.30)

            # Acceleration noise: bias toward moving (mu=0.5) so the car
            # doesn't get stuck at the start line
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(
                a_t_original[0][1],  0.5,  1.00, 0.10)

            # STOCHASTIC BRAKING (Ben Lau's key insight):
            # Only explore braking 10% of the time. This prevents the car
            # from getting stuck in "always brake" local minima while still
            # allowing the network to learn when braking is useful.
            if np.random.random() < 0.1:
                noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(
                    a_t_original[0][2],  0.3,  1.00, 0.10)
            else:
                noise_t[0][2] = 0.0

            a_t[0][0] = np.clip(a_t_original[0][0] + noise_t[0][0], -1.0, 1.0)
            a_t[0][1] = np.clip(a_t_original[0][1] + noise_t[0][1],  0.0, 1.0)
            brake_raw = np.clip(a_t_original[0][2] + noise_t[0][2],  0.0, 1.0)

            # Brake remap:
            # sigmoid heads initialize near 0.5, which otherwise means constant
            # medium braking. Treat 0.5 as "no brake", then scale upper half.
            a_t[0][2] = np.clip((brake_raw - 0.5) * 2.0, 0.0, 1.0)

            # Deadzone to avoid tiny brake drag.
            if a_t[0][2] < 0.05:
                a_t[0][2] = 0.0

            # Never apply throttle and brake together; keep dominant command.
            if a_t[0][1] >= a_t[0][2]:
                a_t[0][2] = 0.0
            else:
                a_t[0][1] = 0.0

            try:
                s_t1, r_t, done, info = env.step(a_t[0])
            except (TimeoutError, OSError, RuntimeError) as e:
                print("\n[step] TORCS error: %s — ending episode early." % e)
                s_t1, r_t, done, info = s_t, -10.0, True, {}

            if info.get('lap_bonus', 0.0) > 0:
                lap_t = info['lastLapTime']
                episode_laps.append(lap_t)
                all_lap_times.append(lap_t)
                if lap_t < best_lap_time:
                    best_lap_time = lap_t

            buff.add(s_t, a_t[0], r_t, s_t1, done)

            batch = buff.getBatch(BATCH_SIZE)
            states     = np.asarray([e[0] for e in batch])
            actions    = np.asarray([e[1] for e in batch])
            rewards    = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones      = np.asarray([e[4] for e in batch])
            y_t        = np.asarray([e[1] for e in batch])

            target_actions   = actor.target_model.predict(new_states,  verbose=0)
            target_q_values  = critic.target_model.predict(
                [new_states, target_actions], verbose=0
            )

            for k in range(len(batch)):
                y_t[k] = rewards[k] if dones[k] else rewards[k] + GAMMA * target_q_values[k]

            if train_indicator:
                loss    += critic.model.train_on_batch([states, actions], y_t)
                a_grads  = actor.model.predict(states, verbose=0)
                grads    = critic.gradients(states, a_grads)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            print("Ep %d | Step %d | Action [%.3f %.3f %.3f] | Reward %.4f | Loss %.6f"
                  % (i, step, a_t[0][0], a_t[0][1], a_t[0][2], r_t, loss))

            step += 1
            if done:
                break

        ep_best = min(episode_laps) if episode_laps else None

        print("── Episode %d complete | total reward: %.2f | laps: %d | best lap: %s"
              % (i, total_reward,
                 len(episode_laps),
                 ("%.3f s" % ep_best) if ep_best else "none"))

        episode_stats.append({
            "episode":       int(i),
            "total_reward":  float(total_reward),
            "n_laps":        len(episode_laps),
            "best_lap":      float(ep_best) if ep_best else None,
            "lap_times":     [float(x) for x in episode_laps],
            "train":         int(train_indicator),
        })

        if i % 3 == 0 and train_indicator:
            print("  Saving weights …")
            actor.model.save_weights(actor_weights, overwrite=True)
            with open(os.path.join(artifact_dir, "actor_laptime.json"), "w") as f:
                f.write(actor.model.to_json())
            critic.model.save_weights(critic_weights, overwrite=True)
            with open(os.path.join(artifact_dir, "critic_laptime.json"), "w") as f:
                f.write(critic.model.to_json())

    env.end()

    stats_path = os.path.join(artifact_dir, "episode_stats_%s.json" % run_tag)
    with open(stats_path, "w") as f:
        json.dump(episode_stats, f, indent=2)
    print("\nWrote episode stats → %s" % stats_path)

    lap_summary = {
        "best_lap_time_s":   float(best_lap_time) if best_lap_time < float('inf') else None,
        "all_lap_times_s":   [float(x) for x in all_lap_times],
        "n_laps_completed":  len(all_lap_times),
        "episodes":          episode_count,
        "train":             int(train_indicator),
    }
    lap_path = os.path.join(artifact_dir, "lap_summary_%s.json" % run_tag)
    with open(lap_path, "w") as f:
        json.dump(lap_summary, f, indent=2)
    print("Wrote lap summary    → %s" % lap_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [e["episode"] for e in episode_stats]
        ys = [e["total_reward"] for e in episode_stats]
        if xs:
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, linewidth=1)
            plt.xlabel("Episode"); plt.ylabel("Total reward")
            plt.title("DDPG TORCS – reward (%s)" % run_tag)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(artifact_dir, "reward_curve_%s.png" % run_tag)
            plt.savefig(p, dpi=120); plt.close()
            print("Wrote reward plot    → %s" % p)

        if all_lap_times:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, len(all_lap_times) + 1), all_lap_times,
                     marker='o', markersize=3, linewidth=1)
            best_line = min(all_lap_times)
            plt.axhline(best_line, color='r', linestyle='--',
                        label="Best: %.3f s" % best_line)
            plt.xlabel("Lap #"); plt.ylabel("Lap time (s)")
            plt.title("Lap times over training (%s)" % run_tag)
            plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            p = os.path.join(artifact_dir, "lap_times_%s.png" % run_tag)
            plt.savefig(p, dpi=120); plt.close()
            print("Wrote lap time plot  → %s" % p)
    except Exception as ex:
        print("Skipping plots:", ex)

    print("\nFinished.  Best lap time: %s s"
          % (("%.3f" % best_lap_time) if best_lap_time < float('inf') else "n/a (no lap completed)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG fastest-lap agent for TORCS")
    parser.add_argument("--train",       type=int,   default=1,         choices=[0, 1])
    parser.add_argument("--episodes",    type=int,   default=500)
    parser.add_argument("--max-steps",   type=int,   default=100000,    dest="max_steps")
    parser.add_argument("--artifact-dir",type=str,   default=".",       dest="artifact_dir")
    parser.add_argument("--run-tag",     type=str,   default="laptime", dest="run_tag")
    parser.add_argument("--tag",         type=str,   dest="run_tag")
    args = parser.parse_args()

    playGame(
        train_indicator=args.train,
        episode_count=args.episodes,
        max_steps=args.max_steps,
        artifact_dir=args.artifact_dir,
        run_tag=args.run_tag,
    )
