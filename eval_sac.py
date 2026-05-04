import os
import json
import argparse
import numpy as np
from gym_torcs_sac import TorcsEnv
from sac import SAC, flatten_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="artifacts/sac_best_model")
    parser.add_argument("--mode", type=str, choices=["baseline", "conditioned"], default="conditioned", help="Run baseline lambda=0 evaluation or 33-dim conditioned")
    args = parser.parse_args()

    eval_budgets = [10.0]
    num_episodes = 10
    max_steps = 100000

    # For baseline lambda=0, we still use the same env but we might ignore budget penalties if we wanted,
    # but the environment now has a hardcoded -0.3 penalty. The baseline comparison mentioned in the prompt
    # implies evaluating the SAME environment (which has the penalty and exhaustion) but with a DIFFERENT model 
    # (one trained with lambda=0 vs the new budget-aware one). Wait, the new one has b_t in state. 
    # The prompt said: "Then compare against your λ=0 baseline on the same budgets. The story you're telling with that table: Loose budget... Tight budget..."
    # So this script evaluates whatever model is passed to it on the different budgets.

    env = TorcsEnv(vision=False, throttle=True, gear_change=False, max_laps=1)
    
    state_dim = 32 if args.mode == "baseline" else 33
    action_dim = 3
    agent = SAC(state_dim, action_dim)
    
    model_path = "artifacts/baseline_model" if args.mode == "baseline" else args.model
    if os.path.exists(model_path + "_actor.pth"):
        agent.load(model_path)
        print(f"Loaded model from {model_path} in {args.mode} mode")
    else:
        print(f"Model {model_path} not found! Evaluation will use random weights.")

    results = []

    for budget in eval_budgets:
        print(f"\n--- Evaluating Budget: {budget} ---")
        lap_completions = 0
        lap_times = []
        energy_used_list = []
        violations = 0
        progresses = []
        termination_reasons = {
            'lap_complete': 0,
            'fuel_exhaustion': 0,  # matches gym_torcs_sac.py last_termination_reason string
            'off_track': 0,
            'spin_out': 0,
            'stuck': 0,
            'timeout': 0,
            'backward': 0,
            'unknown': 0
        }

        for ep in range(num_episodes):
            obs = env.reset(relaunch=(ep==0), budget=budget) # Only relaunch torcs on first episode to save time, or maybe every time for stability?
            
            state = flatten_state(obs)
            done = False
            last_info = None
            
            for t in range(max_steps):
                if args.mode == "baseline":
                    action = agent.select_action(state[:32], evaluate=True)
                else:
                    action = agent.select_action(state, evaluate=True)
                obs, reward, done, info = env.step(action)
                state = flatten_state(obs)
                last_info = info
                
                if done:
                    break
            
            lap_completed = last_info.get("lap_completed", False) if last_info else False
            lap_time = last_info.get("lap_time_sec", None) if last_info else None
            
            if last_info:
                fuel_consumed = last_info.get("fuel_consumed", 0.0)
                fuel_remaining = budget - fuel_consumed
            else:
                fuel_consumed = 0.0
                fuel_remaining = 0.0
                
            track_progress = last_info.get("track_progress", 0.0) if last_info else 0.0
            
            if lap_completed:
                lap_completions += 1
                if lap_time is not None:
                    lap_times.append(lap_time)
            
            energy_used_list.append(fuel_consumed)
            progresses.append(track_progress if not lap_completed else 1.0)
            
            if fuel_remaining <= 0:
                violations += 1

            if last_info:
                reason = last_info.get("termination_reason")
                if not reason:
                    reason = "timeout" if t >= max_steps - 1 else "unknown"
            else:
                reason = "unknown"
                
            if reason not in termination_reasons:
                termination_reasons[reason] = 0
            termination_reasons[reason] += 1
            
            status = 'Finished' if lap_completed else 'DNF'
            print(f" Ep {ep+1}/{num_episodes}: {status} | Time: {lap_time if lap_time else 'N/A'} | Fuel: {fuel_consumed:.2f}/{budget:.1f} | Violation: {fuel_remaining <= 0} | Reason: {reason}")

        comp_rate = lap_completions / num_episodes
        mean_time = np.mean(lap_times) if lap_times else None
        std_time = np.std(lap_times) if lap_times else None
        mean_energy = np.mean(energy_used_list)
        viol_rate = violations / num_episodes
        mean_prog = np.mean([p for p in progresses if p < 1.0]) if any(p < 1.0 for p in progresses) else 1.0

        budget_res = {
            "budget": budget,
            "completion_rate": comp_rate,
            "mean_lap_time": float(mean_time) if mean_time else None,
            "std_lap_time": float(std_time) if std_time else None,
            "mean_energy_used": float(mean_energy),
            "violation_rate": viol_rate,
            "mean_progress_dnf": float(mean_prog),
            "termination_reasons": termination_reasons
        }
        results.append(budget_res)
        print(f"Results for budget {budget}: {budget_res}")
        
    env.end()

    with open(f"eval_results_{args.mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()
