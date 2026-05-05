import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import random
import json
from collections import deque
from gym_torcs_sac import TorcsEnv
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scalar(x):
    return float(np.asarray(x, dtype=np.float32).reshape(-1)[0])


def clip_action(action):
    clipped = np.asarray(action, dtype=np.float32).copy()
    clipped[0] = np.clip(clipped[0], -1.0, 1.0)
    clipped[1] = np.clip(clipped[1], 0.0, 1.0)
    clipped[2] = np.clip(clipped[2], 0.0, 1.0)
    return clipped


def blend_actions(guide_action, policy_action, policy_mix):
    policy_mix = float(np.clip(policy_mix, 0.0, 1.0))
    return clip_action(((1.0 - policy_mix) * guide_action) + (policy_mix * policy_action))


def heuristic_action(obs):
    """
    Track-following control prior adapted from TORCS SimpleDriver/snakeoil.

    The default training horizon in this project is short, so a safe guide
    controller keeps early rollouts on-track long enough for SAC to get
    meaningful replay data instead of learning exclusively from DNFs.
    """
    angle = _scalar(obs.angle)            # normalized by pi in make_observaton()
    angle_rad = angle * np.pi
    track_pos = _scalar(obs.trackPos)     # centered at 0, roughly [-1, 1]
    speed_kmh = max(0.0, _scalar(obs.speedX) * 300.0)
    track = np.asarray(obs.track, dtype=np.float32).reshape(-1)
    center_idx = len(track) // 2 if track.size else 0
    front_sensor = float(track[center_idx]) if track.size else 0.0
    left_sensor = float(track[max(center_idx - 1, 0)]) if track.size else 0.0
    right_sensor = float(track[min(center_idx + 1, track.size - 1)]) if track.size else 0.0

    # SimpleDriver-inspired steering: follow the track axis and recenter, then
    # damp the command at speed to avoid snap over-rotation in long episodes.
    steer_lock = 0.785398
    target_angle = angle_rad - (track_pos * 0.55)
    if speed_kmh > 80.0:
        steer = target_angle / (steer_lock * max(speed_kmh - 80.0, 1.0))
    else:
        steer = target_angle / steer_lock
    steer = float(np.clip(steer, -1.0, 1.0))

    # Estimate corner severity from the forward-facing sensors. This gives the
    # guide a way to back off before the heading error explodes into a spin-out.
    target_speed = 105.0
    if track.size >= 11 and abs(track_pos) < 1.0:
        sin5 = 0.08716
        cos5 = 0.99619
        max_speed_dist = 90.0
        max_speed = 120.0

        if front_sensor > max_speed_dist or (front_sensor >= left_sensor and front_sensor >= right_sensor):
            target_speed = max_speed
        else:
            if right_sensor > left_sensor:
                h = front_sensor * sin5
                b = right_sensor - (front_sensor * cos5)
            else:
                h = front_sensor * sin5
                b = left_sensor - (front_sensor * cos5)
            sin_angle = (b * b) / max((h * h) + (b * b), 1e-6)
            target_speed = max_speed * (front_sensor * sin_angle / max_speed_dist)

    target_speed = float(np.clip(target_speed, 35.0, 120.0))
    target_speed -= 30.0 * min(abs(track_pos), 1.0)
    target_speed -= 55.0 * min(abs(angle_rad), 0.7)
    if front_sensor < 40.0:
        target_speed = min(target_speed, 55.0)
    if abs(track_pos) > 0.65:
        target_speed = min(target_speed, 50.0)
    target_speed = float(np.clip(target_speed, 30.0, 120.0))

    speed_error = target_speed - speed_kmh
    accel_brake = (2.0 / (1.0 + np.exp((speed_kmh - target_speed) / 12.0))) - 1.0
    if accel_brake >= 0.0:
        accel = float(np.clip(accel_brake, 0.0, 1.0))
        brake = 0.0
    else:
        accel = 0.0
        brake = float(np.clip(-accel_brake, 0.0, 1.0))

    if speed_kmh < 20.0:
        accel = max(accel, 0.55)
        brake = 0.0

    if abs(angle_rad) > 0.30:
        accel = min(accel, 0.20)
        brake = max(brake, 0.30)

    if abs(track_pos) > 0.80:
        accel = min(accel, 0.15)
        brake = max(brake, 0.35)

    if speed_error < -20.0:
        brake = max(brake, min(0.8, (-speed_error) / 45.0))
        accel = min(accel, 0.10)

    return clip_action(np.array([steer, accel, brake], dtype=np.float32))


def policy_mix_from_buffer_size(buffer_size, warmup_steps, ramp_steps, max_mix=1.0):
    mix = np.clip((buffer_size - warmup_steps) / max(float(ramp_steps), 1.0), 0.0, 1.0)
    return float(np.clip(mix * max_mix, 0.0, 1.0))


def checkpoint_score(summary):
    lap_time_sec = summary.get("lap_time_sec")
    finished = bool(summary.get("finished", False))
    lap_speed_score = 0.0
    if finished and lap_time_sec:
        lap_speed_score = 1.0 / max(float(lap_time_sec), 1.0)

    termination_reason = summary.get("termination_reason")
    termination_rank = {
        "lap_complete": 4,
        "race_finished_socket_closed": 4,
        "max_steps": 3,
        "off_track": 2,
        "spin_out": 1,
        "backward": 0,
        "stuck": -1,
    }.get(termination_reason, 0)

    return (
        1 if finished else 0,
        int(summary.get("lap_count", 0)),
        termination_rank,
        float(summary.get("episode_distance_raced_m", 0.0)),
        int(summary.get("steps_taken", 0)),
        lap_speed_score,
        float(summary.get("reward_total", -np.inf)),
    )

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

def flatten_state(obs):
    features = np.hstack((
        obs.angle, 
        obs.track, 
        obs.trackPos, 
        obs.speedX, 
        obs.speedY, 
        obs.speedZ, 
        obs.wheelSpinVel/100.0, 
        obs.rpm,
        obs.fuelLevel,
        obs.fuelConsumed,
        obs.fuelBudgetRemaining
    ))
    return np.array(features, dtype=np.float32)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.bn0 = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 600)
        
        self.mu = nn.Linear(600, action_dim)
        self.log_std = nn.Linear(600, action_dim)
        
        # Initialize biases to encourage forward movement at start
        # action_dim is 3: (steer, accel, brake)
        with torch.no_grad():
            self.mu.bias.data.copy_(torch.tensor([0.0, 1.0, -2.0]))
            self.mu.weight.data.fill_(0.0)
        
    def forward(self, state):
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Scaling actions: Steer [-1,1], Accel [0,1], Brake [0,1]
        steer = action[:, 0:1]
        accel = (action[:, 1:2] + 1) / 2.0
        brake = (action[:, 2:3] + 1) / 2.0
        scaled_action = torch.cat([steer, accel, brake], dim=1)
        
        # For evaluation (mean action)
        mean_y = torch.tanh(mu)
        mean_steer = mean_y[:, 0:1]
        mean_accel = (mean_y[:, 1:2] + 1) / 2.0
        mean_brake = (mean_y[:, 2:3] + 1) / 2.0
        mean_scaled = torch.cat([mean_steer, mean_accel, mean_brake], dim=1)
        
        return scaled_action, log_prob, mean_scaled

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.bn1 = nn.BatchNorm1d(state_dim)
        self.fc1 = nn.Linear(state_dim + action_dim, 300)
        self.fc2 = nn.Linear(300, 600)
        self.fc3 = nn.Linear(600, 1)

        # Q2 architecture
        self.bn2 = nn.BatchNorm1d(state_dim)
        self.fc4 = nn.Linear(state_dim + action_dim, 300)
        self.fc5 = nn.Linear(300, 600)
        self.fc6 = nn.Linear(600, 1)

    def forward(self, state, action):
        s1 = self.bn1(state)
        xu1 = torch.cat([s1, action], 1)
        x1 = F.relu(self.fc1(xu1))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)

        s2 = self.bn2(state)
        xu2 = torch.cat([s2, action], 1)
        x2 = F.relu(self.fc4(xu2))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)

        return q1, q2

class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.target_entropy = -4.0  # Fix 4: lower than -action_dim (-3.0) for more deterministic racing policy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=1e-4)

        self.gamma = 0.99
        self.tau = 0.001

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        self.actor.train()
        return action.detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=128, behavior_cloning_weight=0.0):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.log_alpha.exp() * next_log_prob
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.log_alpha.exp() * log_prob) - min_q_pi).mean()
        if behavior_cloning_weight > 0.0:
            actor_loss = actor_loss + (behavior_cloning_weight * F.mse_loss(pi, action))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pth")
        torch.save(self.log_alpha.detach().cpu(), filename + "_log_alpha.pth")
        
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        critic_target_path = filename + "_critic_target.pth"
        if os.path.exists(critic_target_path):
            self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())

        log_alpha_path = filename + "_log_alpha.pth"
        if os.path.exists(log_alpha_path):
            loaded_log_alpha = torch.load(log_alpha_path, map_location=device)
            self.log_alpha.data.copy_(loaded_log_alpha.to(device))


def load_checkpoint_metadata(filename):
    metadata_path = filename + "_meta.json"
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint_metadata(filename, summary, score):
    metadata_path = filename + "_meta.json"
    payload = {
        "summary": summary,
        "score": list(score),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--artifact-dir", type=str, default="./artifacts")
    parser.add_argument("--run-tag", type=str, default="sac")
    parser.add_argument("--model-prefix", type=str, default="sac_best_model")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--fuel-lambda", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    env = TorcsEnv(
        vision=False,
        throttle=True,
        gear_change=False,
        fuel_lambda=args.fuel_lambda,
        max_laps=1,
    )
    
    state_dim = 32
    action_dim = 3
    batch_size = 64
    guided_warmup_steps = 384
    min_replay_to_train = 128
    policy_ramp_steps = 4000
    train_noise_scale = np.array([0.02, 0.05, 0.03], dtype=np.float32)
    eval_policy_mix = 0.05
    behavior_cloning_weight = 0.05
    train_policy_mix_cap = 0.20
    finisher_policy_mix_cap = 0.60

    agent = SAC(state_dim, action_dim)
    replay_buffer = ReplayBuffer(100000)
    episode_stats = []
    stats_path = os.path.join(args.artifact_dir, f"episode_stats_{args.run_tag}.json")

    best_reward = -np.inf
    best_model_score = None
    model_path = os.path.join(args.artifact_dir, args.model_prefix)
    checkpoint_exists = os.path.exists(model_path + "_actor.pth") and os.path.exists(model_path + "_critic.pth")
    checkpoint_metadata = load_checkpoint_metadata(model_path) if checkpoint_exists else None

    checkpoint_loaded = False
    if checkpoint_exists and args.resume:
        try:
            agent.load(model_path)
            checkpoint_loaded = True
            print(f"Loaded model checkpoint from {model_path}")
        except Exception as exc:
            print(f"Failed to load checkpoint from {model_path}: {exc}")
            print("Starting from fresh weights instead.")
            checkpoint_exists = False
    elif not checkpoint_exists and not args.train:
        print(f"No trained model found at {model_path}, running with random weights.")
    elif checkpoint_exists and not args.resume:
        print(f"Checkpoint exists at {model_path}, but resume is disabled. Starting fresh.")

    best_checkpoint_finished = bool(
        checkpoint_metadata
        and checkpoint_metadata.get("summary", {}).get("finished", False)
    )
    effective_eval_policy_mix = eval_policy_mix if best_checkpoint_finished else 0.0
    saw_training_finish = False

    for ep in range(args.episodes):
        obs = env.reset(relaunch=True)
        state = flatten_state(obs)
        ep_reward = 0
        episode_last_info = None
        peak_speed_kmh = 0.0
        fuel_penalty_sum = 0.0
        max_fuel_penalty = 0.0
        max_fuel_over_consumption = 0.0
        min_fuel_budget_remaining = float("inf")
        
        for t in range(args.max_steps):
            guide_action = heuristic_action(obs)
            if args.train:
                if len(replay_buffer) < guided_warmup_steps:
                    action = guide_action
                    noise_scale = train_noise_scale
                else:
                    policy_action = agent.select_action(state, evaluate=False)
                    current_mix_cap = finisher_policy_mix_cap if saw_training_finish else train_policy_mix_cap
                    policy_mix = policy_mix_from_buffer_size(
                        len(replay_buffer),
                        guided_warmup_steps,
                        policy_ramp_steps,
                        max_mix=current_mix_cap,
                    )
                    action = blend_actions(guide_action, policy_action, policy_mix)
                    noise_scale = train_noise_scale * max(0.15, 1.0 - policy_mix)
                action = clip_action(action + np.random.normal(0.0, noise_scale, size=3))
            else:
                if checkpoint_loaded:
                    policy_action = agent.select_action(state, evaluate=True)
                    action = blend_actions(guide_action, policy_action, effective_eval_policy_mix)
                else:
                    action = guide_action
                
            obs, reward, done, info = env.step(action)
            next_state = flatten_state(obs)
            
            if args.train:
                # Store unscaled action if needed, but our buffer stores scaled action
                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) >= min_replay_to_train:
                    agent.train(
                        replay_buffer,
                        batch_size=min(batch_size, len(replay_buffer)),
                        behavior_cloning_weight=behavior_cloning_weight,
                    )
                    
            state = next_state
            ep_reward += reward
            episode_last_info = info
            peak_speed_kmh = max(peak_speed_kmh, float(info.get("speedX_kmh", 0.0)))
            fuel_penalty = float(info.get("fuel_penalty", 0.0))
            fuel_penalty_sum += fuel_penalty
            max_fuel_penalty = max(max_fuel_penalty, fuel_penalty)
            max_fuel_over_consumption = max(
                max_fuel_over_consumption,
                float(info.get("fuel_over_consumption", 0.0)),
            )
            min_fuel_budget_remaining = min(
                min_fuel_budget_remaining,
                float(info.get("fuel_budget_remaining", 0.0)),
            )
            
            if done:
                break

        if episode_last_info is None:
            episode_last_info = {
                "done": False,
                "lap_completed": False,
                "lap_count": 0,
                "lap_time_sec": None,
                "termination_reason": None,
                "fuel_lambda": float(args.fuel_lambda),
                "fuel_current": 0.0,
                "fuel_consumed": 0.0,
                "fuel_penalty": 0.0,
                "fuel_rate_avg": 0.0,
                "fuel_budget_rate_target": 0.0,
                "fuel_over_consumption": 0.0,
                "fuel_budget_remaining": 0.0,
                "track_progress": 0.0,
                "speedX_kmh": 0.0,
            }

        steps_taken = t + 1
        lap_completed = bool(episode_last_info.get("lap_completed", False))
        termination_reason = episode_last_info.get("termination_reason")
        if termination_reason is None:
            termination_reason = "max_steps" if steps_taken >= args.max_steps else "unknown"
        dnf = not lap_completed
        episode_summary = {
            "episode": int(ep + 1),
            "train": int(args.train),
            "run_tag": args.run_tag,
            "reward_total": float(ep_reward),
            "steps_taken": int(steps_taken),
            "finished": lap_completed,
            "dnf": bool(dnf),
            "lap_count": int(episode_last_info.get("lap_count", 0)),
            "lap_time_sec": (
                float(episode_last_info["lap_time_sec"])
                if episode_last_info.get("lap_time_sec") is not None
                else None
            ),
            "termination_reason": termination_reason,
            "fuel_lambda": float(episode_last_info.get("fuel_lambda", args.fuel_lambda)),
            "fuel_current": float(episode_last_info.get("fuel_current", 0.0)),
            "fuel_consumed": float(episode_last_info.get("fuel_consumed", 0.0)),
            "fuel_penalty_last": float(episode_last_info.get("fuel_penalty", 0.0)),
            "fuel_penalty_sum": float(fuel_penalty_sum),
            "fuel_rate_avg_last": float(episode_last_info.get("fuel_rate_avg", 0.0)),
            "fuel_budget_rate_target": float(episode_last_info.get("fuel_budget_rate_target", 0.0)),
            "fuel_over_consumption_last": float(episode_last_info.get("fuel_over_consumption", 0.0)),
            "fuel_over_consumption_max": float(max_fuel_over_consumption),
            "fuel_budget_remaining_last": float(episode_last_info.get("fuel_budget_remaining", 0.0)),
            "fuel_budget_remaining_min": float(
                min_fuel_budget_remaining if min_fuel_budget_remaining != float("inf") else 0.0
            ),
            "episode_distance_raced_m": float(episode_last_info.get("episode_distance_raced_m", 0.0)),
            "track_progress_last": float(episode_last_info.get("track_progress", 0.0)),
            "peak_speed_kmh": float(peak_speed_kmh),
            "final_speed_kmh": float(episode_last_info.get("speedX_kmh", 0.0)),
            "max_fuel_penalty": float(max_fuel_penalty),
            "replay_size": int(len(replay_buffer)),
            "best_reward_so_far": float(max(best_reward, ep_reward)),
        }
        episode_stats.append(episode_summary)
        best_reward = max(best_reward, ep_reward)
        saw_training_finish = saw_training_finish or lap_completed

        with open(stats_path, "w", encoding="utf-8") as sf:
            json.dump(episode_stats, sf, indent=2)

        status = "FINISH" if lap_completed else "DNF"
        lap_time_str = (
            f"{episode_summary['lap_time_sec']:.1f}s"
            if episode_summary["lap_time_sec"] is not None
            else "n/a"
        )
        print(
            f"[{args.run_tag}] Episode: {ep+1}/{args.episodes} | {status} | "
            f"Reward: {ep_reward:.2f} | Steps: {steps_taken} | "
            f"Lap: {lap_time_str} | Reason: {termination_reason} | "
            f"FuelLambda: {episode_summary['fuel_lambda']:.3f}"
        )
        
        if args.train:
            model_score = checkpoint_score(episode_summary)
            if best_model_score is None or model_score > best_model_score:
                best_model_score = model_score
                distance_m = float(episode_summary["episode_distance_raced_m"])
                finish_note = "finish" if lap_completed else f"{distance_m:.0f}m raced"
                agent.save(model_path)
                save_checkpoint_metadata(model_path, episode_summary, model_score)
                print(
                    f"Saved new best model with reward: {ep_reward:.2f} "
                    f"({finish_note})"
                )
            
    env.end()

if __name__ == "__main__":
    main()