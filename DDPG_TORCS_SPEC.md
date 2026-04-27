# TORCS DDPG Autonomous Racing Specification

## 1. Problem Definition
The objective is to train a Deep Reinforcement Learning agent using the **Deep Deterministic Policy Gradient (DDPG)** algorithm to autonomously drive a race car in the **TORCS (The Open Racing Car Simulator)** environment. 

Specifically, we are optimizing for performance on the **alpine-1** track. The agent must learn to:
- Maximize forward velocity and progress along the track.
- Minimize lap times through optimal racing lines and throttle management.
- Avoid collisions with walls and track boundaries.
- Maintain vehicle stability (avoiding spins and burnouts).
- Recover from or avoid "stuck" states where no progress is being made.

---

## 2. State Space (Observations)
The agent receives a **29-dimensional** continuous state vector representing the current physical state of the vehicle and its relation to the track:

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

---

## 3. Action Space
The agent outputs a **3-dimensional** continuous action vector:

1.  **Steering**: `[-1.0, 1.0]` (Full Right to Full Left).
2.  **Acceleration**: `[0.0, 1.0]` (No throttle to Full throttle).
3.  **Brake**: `[0.0, 1.0]` (No braking to Full braking).

---

## 4. Reward Framework
The reward function is meticulously tuned to balance speed against safety.

### 4.1 Positive Rewards (Incentives)
-   **Forward Progress**: `(SpeedX * cos(Angle) - abs(SpeedX * sin(Angle))) / 50.0`. This rewards moving forward along the track axis while penalizing lateral sliding.
-   **Top Speed Bonuses**:
    -   Speed > 130 km/h: **+15.0** per step
    -   Speed > 120 km/h: **+10.0** per step
    -   Speed > 110 km/h: **+5.0** per step
    -   Speed > 90 km/h: **+1.0** per step
-   **Lap Completion Payday**: `50000.0 / LapTime`. Finishing a lap triggers a massive bonus. A 300s lap yields **~+166**, whereas a 150s lap yields **~+333**.

### 4.2 Negative Rewards (Penalties)
-   **Collision/Damage**: `-(DamageDelta / 2.0)`. A hard hit is heavily penalized.
-   **Track Centering**: `-0.1 * abs(TrackPos)`. Small penalty for drifting from the center line.
-   **Wall Hugging**: **-1.0** per step if `abs(TrackPos) > 0.95`. Prevents "scraping" along walls.
-   **Spin Penalty**: **-2.0** per step if `abs(Angle) > 0.5` (~30 degrees). Discourages losing control.
-   **Backward Driving**: **-5.0** per step if `SpeedX < 0`.
-   **Stuck Termination**: **-500.0** flat penalty if speed remains below 5 km/h for 100 consecutive steps.

---

## 5. Training Configuration
-   **Episode Length**: 3 Laps (The episode terminates and saves weights after completing 3 full laps).
-   **Exploration**: Ornstein-Uhlenbeck noise process added to actions, decaying over 100,000 steps.
-   **Saving Cadence**: Model weights are saved to disk at the end of **every episode**.
-   **Relaunch Strategy**: TORCS is relaunched at the start of every episode to ensure environment stability and reset the GUI automation.

---

## 6. Current Solution Architecture (DDPG)

The project uses a **Deep Deterministic Policy Gradient (DDPG)** architecture, consisting of an Actor and a Critic.

### 6.1 Actor Network
The Actor maps the 29 sensors directly to the best-known continuous actions.
-   **Input Layer**: 29 units (normalized sensors).
-   **Hidden Layer 1**: 300 units, ReLU activation.
-   **Hidden Layer 2**: 600 units, ReLU activation.
-   **Output Layer**:
    -   **Steering**: 1 unit, Tanh activation (Range: [-1, 1]).
    -   **Acceleration**: 1 unit, Sigmoid activation (Range: [0, 1]).
    -   **Brake**: 1 unit, Sigmoid activation (Range: [0, 1]).

### 6.2 Critic Network
The Critic estimates the Q-value (expected future reward) for a given State-Action pair. It uses a **Late-Fusion** architecture:
-   **State Pathway**: 300 units (ReLU) -> 600 units (Linear).
-   **Action Pathway**: 600 units (Linear).
-   **Fusion**: The State and Action pathways are **Added** together.
-   **Post-Fusion**: 600 units, ReLU activation.
-   **Output**: 3 units (Linear), representing the estimated value gradient for the action components.

### 6.3 Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Number of samples per training step. |
| **Buffer Size** | 100,000 | Experience Replay memory capacity. |
| **Gamma ($\gamma$)** | 0.99 | Discount factor for future rewards. |
| **Tau ($\tau$)** | 0.001 | Soft update rate for target networks. |
| **Learning Rate (Actor)** | 0.0001 | Optimizer step size for the Actor. |
| **Learning Rate (Critic)** | 0.001 | Optimizer step size for the Critic. |
| **Exploration (Steps)** | 100,000 | Total steps over which noise decays. |

