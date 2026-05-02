# TORCS SAC Autonomous Racing Specification

## 1. Problem Definition
The objective is to train a Deep Reinforcement Learning agent using the **Soft Actor-Critic (SAC)** algorithm to autonomously drive a race car in the **TORCS (The Open Racing Car Simulator)** environment. 

Specifically, we are optimizing for performance on the **aalborg** track, with an added constraint on **fuel efficiency**. The agent must learn to:
- Maximize forward velocity and progress along the track.
- Minimize lap times through optimal racing lines and throttle management.
- Complete the race within a strict fuel budget using a Lagrangian soft-penalty mechanism.
- Avoid collisions with walls and track boundaries to survive the episode.
- Recover from or avoid "stuck" states where no progress is being made.

---

## 2. State Space (Observations)
The agent receives a **32-dimensional** continuous state vector representing the current physical state of the vehicle and its relation to the track, augmented with fuel constraint information:

| Sensor Type | Dimension | Description |
| :--- | :--- | :--- |
| **Angle** | 1 | Angle between the car direction and the track axis. |
| **Track** | 19 | Distance to the track edge in 19 directions (200m range). |
| **TrackPos** | 1 | Distance from the track center: 0 is center, [-1, 1] is edge. |
| **SpeedX** | 1 | Longitudinal velocity (forward speed). |
| **SpeedY** | 1 | Lateral velocity (side slip). |
| **SpeedZ** | 1 | Vertical velocity. |
| **WheelSpinVel** | 4 | Rotation speed of the 4 wheels. |
| **RPM** | 1 | Engine Revolutions Per Minute. |
| **FuelLevel** | 1 | Current fuel remaining in the tank. |
| **FuelConsumed** | 1 | Total fuel consumed since the start of the race. |
| **FuelBudgetRemaining**| 1 | Prospective ratio of remaining fuel budget versus remaining lap progress, clamped to `[0, 5]`. |

---

## 3. Action Space
The agent outputs a **3-dimensional** continuous action vector:

1.  **Steering**: `[-1.0, 1.0]` (Full Right to Full Left).
2.  **Acceleration**: `[0.0, 1.0]` (No throttle to Full throttle).
3.  **Brake**: `[0.0, 1.0]` (No braking to Full braking).

---

## 4. Reward Framework
The reward function is meticulously tuned to balance speed against safety and fuel efficiency, with strict failure states to prevent the agent from exploiting stuck conditions.

### 4.1 Positive Rewards (Incentives)
-   **Forward Progress**: `(SpeedX * cos(Angle) - abs(SpeedX * sin(Angle))) / 5.0`. This is a strong linear reward for moving forward along the track axis while penalizing lateral sliding. Driving at 50 km/h yields ~10.0 reward per step.

### 4.2 Negative Rewards (Penalties)
-   **Fuel Lagrangian Penalty**: `-lambda * max(0, fuel_consumed_rate_avg - budget_rate_target) / budget_rate_target`. A soft penalty that only activates when the rolling average fuel burn exceeds the target budget rate. `lambda` is configurable at runtime through `--fuel-lambda`.
-   **Collision/Damage**: `-(DamageDelta / 10.0)`. A softened penalty for scraping or hitting walls.
-   **Minimum-Speed Pressure**: **-0.5** per step if `SpeedX < 10 km/h`. This discourages the degenerate "accelerate briefly then stop" policy without forcing termination.

### 4.3 Terminal Failure States
Instead of paralyzing step-by-step penalties for driving poorly, the agent faces strict episode termination conditions if it makes a critical error. This forces it to learn to drive safely to maximize the duration of the episode.
-   **Out of Track**: **-200.0** flat penalty and episode termination if `abs(TrackPos) > 0.999`.
-   **Spin Out**: **-200.0** flat penalty and episode termination if `abs(Angle) > 1.0`.
-   **Stuck**: **-200.0** flat penalty and episode termination if the car advances less than `8.0m` over a `100`-step window.
-   **Backward Driving**: **-200.0** flat penalty and episode termination if `cos(Angle) < 0`.

### 4.4 Positive Milestone Rewards
-   **Waypoint / Sector Bonus**: every 500m of forward progress, the agent receives a bonus based on how quickly it reached that waypoint relative to a target of 900 steps.
-   **Lap Completion Bonus**: `50000.0 / LapTime`. Completing the lap yields a large bonus scaled inversely by lap time.

---

## 5. Training Configuration
-   **Episode Length**: 1 Lap. The environment terminates when the configured lap quota is reached.
-   **Max Steps**: controlled by the SAC CLI via `--max-steps` and currently defaults to `1000000` in `sac.py`. Practical runs can override this externally.
-   **Exploration**: Entropy-based stochastic policy. The agent samples actions from a squashed Gaussian distribution, automatically balancing exploration and exploitation. The first 1000 steps use random warmup actions biased toward forward acceleration to seed the replay buffer.
-   **Saving Cadence**: Model weights are saved to disk whenever a new "best" episode reward is achieved.
-   **Relaunch Strategy**: TORCS is relaunched at the start of every episode to ensure environment stability and reset the GUI automation.
-   **Fuel Budget**: 1-lap budget of `5.0` fuel units, with `FuelBudgetRemaining` computed prospectively from Aalborg track progress using a track length constant of `2598.63m`.
-   **Episode Logging**: SAC writes episode summaries to `episode_stats_<run_tag>.json` and lap completions to `lap_times.log`.

---

## 6. Current Solution Architecture (SAC)

The project uses a **Soft Actor-Critic (SAC)** architecture, an off-policy algorithm that optimizes a stochastic policy in an entropy-regularized reinforcement learning setting. It consists of an Actor and Twin Critics.

### 6.1 Actor Network
The Actor maps the 32 sensors to the mean and standard deviation of a Gaussian distribution for the actions.
-   **Input Layer**: 32 units (with Batch Normalization to stabilize varying sensor scales).
-   **Hidden Layer 1**: 300 units, ReLU activation.
-   **Hidden Layer 2**: 600 units, ReLU activation.
-   **Output Mean ($\mu$) Layer**: 3 units (Linear).
-   **Output Log Std Layer**: 3 units (Linear), clamped between `[-20, 2]`.
-   **Squashing**: The sampled actions are passed through a `Tanh` function to bind them between `[-1, 1]`. Steering remains `[-1, 1]`, while Acceleration and Brake are shifted and scaled to `[0, 1]`.
-   **Initialization Detail**: the mean head is biased toward forward motion at startup with bias `[0.0, 1.0, -2.0]`.

### 6.2 Twin Critic Networks
Two identical Critics independently estimate the soft Q-value for a given State-Action pair. The minimum of the two is used during policy updates to heavily mitigate the overestimation bias that plagues DDPG.
-   **Input Layer**: 35 units (State + Action concatenated).
-   **Hidden Layer 1**: 300 units, ReLU activation.
-   **Hidden Layer 2**: 600 units, ReLU activation.
-   **Output**: 1 unit (Linear), representing the estimated soft Q-value.

### 6.3 Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 128 | Number of samples per training step. |
| **Buffer Size** | 100,000 | Experience Replay memory capacity. |
| **Gamma ($\gamma$)** | 0.99 | Discount factor for future rewards. |
| **Tau ($\tau$)** | 0.001 | Soft update rate for target networks. |
| **Actor LR** | 0.0001 | Optimizer step size for the Actor. |
| **Critic LR** | 0.001 | Optimizer step size for the Twin Critics. |
| **Entropy LR** | 0.0001 | Optimizer step size for automatic temperature tuning. |
| **Target Entropy** | -4.0 | Lower than `-action_dim` to push earlier convergence toward more deterministic control. |
| **Initial Random Steps**| 1000 | Number of random warmup actions to take before utilizing the Actor network; these are biased toward forward acceleration rather than fully uniform. |
| **Fuel Lambda** | CLI-configurable (`--fuel-lambda`) | Lagrangian penalty multiplier for fuel overuse. |
