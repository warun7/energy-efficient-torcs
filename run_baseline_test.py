import numpy as np
from gym_torcs_sac import TorcsEnv
from sac import SAC
import time

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

agent = SAC(state_dim=32, action_dim=3)
agent.load("artifacts/baseline_model")

ob = env.reset(relaunch=True)
state = ob[:-1]

done = False
for step in range(10000):
    action = agent.select_action(state, evaluate=True)
    ob_new, reward, done, info = env.step(action)
    state = ob_new[:-1]
    
    if done:
        print(f"Done! Reason: {info.get('termination_reason')}, Progress: {info.get('track_progress_last')}, Fuel: {info.get('fuel_consumed')}")
        break
