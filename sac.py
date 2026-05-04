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
        obs.fuelBudgetRemaining,
        obs.b_t
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
        # Stable log_prob calculation to prevent gradient explosion:
        # log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        # log_prob -= torch.log(1 - action.pow(2) + 1e-6)  <-- numerically unstable
        import math
        log_prob = dist.log_prob(x_t)
        log_prob -= (2 * (math.log(2) - x_t - F.softplus(-2 * x_t)))
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-5)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-5)

        self.target_entropy = -4.0  # Fix 4: lower than -action_dim (-3.0) for more deterministic racing policy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-5)

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

    def train(self, replay_buffer, batch_size=128):
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
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        pi, log_prob, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.log_alpha.exp() * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
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
        # Helper to load and reshape weights if state dimension changed (e.g., 32 to 33)
        def _load_and_reshape(model, path, is_critic=False):
            if not os.path.exists(path): return
            old_dict = torch.load(path, map_location=device)
            new_dict = model.state_dict()
            for key in old_dict:
                if 'fc1.weight' in key or (is_critic and 'fc4.weight' in key):
                    if old_dict[key].shape[1] == 32 and new_dict[key].shape[1] == 33: # Actor
                        new_dict[key][:, :32] = old_dict[key]
                        new_dict[key][:, 32:] = torch.randn(new_dict[key].shape[0], 1) * 0.01
                    elif old_dict[key].shape[1] == 35 and new_dict[key].shape[1] == 36: # Critic
                        new_dict[key][:, :32] = old_dict[key][:, :32]
                        new_dict[key][:, 32:33] = torch.randn(new_dict[key].shape[0], 1) * 0.01
                        new_dict[key][:, 33:] = old_dict[key][:, 32:]
                    else:
                        new_dict[key] = old_dict[key]
                elif 'bn0' in key or (is_critic and ('bn1' in key or 'bn2' in key)):
                    if len(old_dict[key].shape) > 0 and old_dict[key].shape[0] == 32 and new_dict[key].shape[0] == 33:
                        new_dict[key][:32] = old_dict[key]
                    else:
                        new_dict[key] = old_dict[key]
                else:
                    new_dict[key] = old_dict[key]
            model.load_state_dict(new_dict)

        _load_and_reshape(self.actor, filename + "_actor.pth", is_critic=False)
        _load_and_reshape(self.critic, filename + "_critic.pth", is_critic=True)
        
        critic_target_path = filename + "_critic_target.pth"
        if os.path.exists(critic_target_path):
            _load_and_reshape(self.critic_target, critic_target_path, is_critic=True)
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())

        log_alpha_path = filename + "_log_alpha.pth"
        if os.path.exists(log_alpha_path):
            loaded_log_alpha = torch.load(log_alpha_path, map_location=device)
            self.log_alpha.data.copy_(loaded_log_alpha.to(device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--artifact-dir", type=str, default="./artifacts")
    parser.add_argument("--run-tag", type=str, default="sac")
    parser.add_argument("--model-prefix", type=str, default="sac_best_model")
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--fuel-lambda", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)
    env = TorcsEnv(
        vision=False,
        throttle=True,
        gear_change=False,
        fuel_lambda=args.fuel_lambda,
        max_laps=1,
    )
    
    state_dim = 33
    action_dim = 3

    agent = SAC(state_dim, action_dim)
    replay_buffer = ReplayBuffer(100000)
    episode_stats = []
    stats_path = os.path.join(args.artifact_dir, f"episode_stats_{args.run_tag}.json")

    best_reward = -np.inf
    model_path = os.path.join(args.artifact_dir, args.model_prefix)
    checkpoint_exists = os.path.exists(model_path + "_actor.pth") and os.path.exists(model_path + "_critic.pth")

    if checkpoint_exists and args.resume:
        agent.load(model_path)
        print(f"Loaded model checkpoint from {model_path}")
    elif not checkpoint_exists and not args.train:
        print(f"No trained model found at {model_path}, running with random weights.")
    elif checkpoint_exists and not args.resume:
        print(f"Checkpoint exists at {model_path}, but resume is disabled. Starting fresh.")
    
    # Warm-start logic relies on replay buffer being empty initially
    # ReplayBuffer is already empty on init, so we just don't load any old replays here.

    for ep in range(args.episodes):
        # Biased budget sampling: 40% tight, 40% mid, 20% loose
        # Loose ceiling tightened from 8.0 → 6.5 so agent can't farm easy wins on huge budgets
        r = np.random.random()
        if r < 0.40:
            budget = float(np.random.uniform(3.0, 4.5))   # tight: physically hard, needs coasting
        elif r < 0.80:
            budget = float(np.random.uniform(4.5, 5.5))   # mid: tight but completable with care
        else:
            budget = float(np.random.uniform(5.5, 6.5))   # loose: comfortable margin
        obs = env.reset(relaunch=True, budget=budget)
        print(f"\n--- Starting Episode {ep+1} | Budget: {env.total_fuel_budget:.2f} ({'tight' if budget < 4.5 else 'mid' if budget < 6.5 else 'loose'}) ---")
        state = flatten_state(obs)
        ep_reward = 0
        episode_last_info = None
        peak_speed_kmh = 0.0
        fuel_penalty_sum = 0.0
        worst_fuel_penalty = 0.0   # most negative value seen (tracks worst overconsumption step)
        max_fuel_over_consumption = 0.0
        min_fuel_budget_remaining = float("inf")
        
        for t in range(args.max_steps):
            if args.train:
                if len(replay_buffer) < 1000:
                    # Random actions from space limits
                    # Steer: [-1, 1], Accel: [0, 1], Brake: [0, 1]
                    # Biased to accelerate instead of getting stuck
                    action = np.array([
                        np.random.uniform(-0.1, 0.1),  # steer slightly
                        np.random.uniform(0.5, 1.0),   # mostly accelerate
                        np.random.uniform(0.0, 0.1)    # rarely brake
                    ])
                else:
                    action = agent.select_action(state, evaluate=False)
            else:
                action = agent.select_action(state, evaluate=True)
                
            obs, reward, done, info = env.step(action)
            next_state = flatten_state(obs)
            
            if args.train:
                # Store unscaled action if needed, but our buffer stores scaled action
                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) > 1000:
                    agent.train(replay_buffer, batch_size=128)
                    
            state = next_state
            ep_reward += reward
            episode_last_info = info
            peak_speed_kmh = max(peak_speed_kmh, float(info.get("speedX_kmh", 0.0)))
            fuel_penalty = float(info.get("fuel_penalty", 0.0))
            fuel_penalty_sum += fuel_penalty
            worst_fuel_penalty = min(worst_fuel_penalty, fuel_penalty)  # most negative = worst step
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
                "fuel_budget": 0.0,
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
            "fuel_budget": float(episode_last_info.get("fuel_budget", 0.0)),
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
            "track_progress_last": float(episode_last_info.get("track_progress", 0.0)),
            "peak_speed_kmh": float(peak_speed_kmh),
            "final_speed_kmh": float(episode_last_info.get("speedX_kmh", 0.0)),
            "max_fuel_penalty": float(abs(worst_fuel_penalty)),  # absolute magnitude of worst step penalty
            "worst_fuel_penalty_raw": float(worst_fuel_penalty),  # raw (negative) for debugging
            "replay_size": int(len(replay_buffer)),
            "best_reward_so_far": float(max(best_reward, ep_reward)),
        }
        episode_stats.append(episode_summary)

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
            f"Budget: {episode_summary.get('fuel_budget', 0.0):.2f}"
        )
        
        # Verify b_t neuron is receiving signal
        if (ep + 1) % 10 == 0 and agent.actor.fc1.weight.grad is not None:
            bt_grad = agent.actor.fc1.weight.grad[:, 32].abs().mean().item()
            old_grad = agent.actor.fc1.weight.grad[:, :32].abs().mean().item()
            ratio = bt_grad / old_grad if old_grad > 0 else 0.0
            print(f"b_t gradient ratio: {ratio:.4f} (bt_grad: {bt_grad:.6f}, old_grad: {old_grad:.6f})")
        
        if args.train and ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(model_path)
            print(f"Saved new best model with reward: {best_reward:.2f}")
            
    env.end()

if __name__ == "__main__":
    main()
