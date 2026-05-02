# SAC on TORCS: First-Principles Deep Dive

This document explains your `sac.py` and `gym_torcs_sac.py` implementation from the ground up.
It is written for presentation prep, with both intuition and implementation-level detail.

---

## 1) The Problem From First Principles

### 1.1 What are we trying to solve?

You want an autonomous driving policy that controls a race car in TORCS and learns by trial-and-error.
At each time step, the agent observes the current state, picks an action, gets a reward, and transitions to a new state.
The objective is to maximize long-term cumulative reward, not just immediate reward.

Formally, this is a **Markov Decision Process (MDP)**:

- **State** `s_t`: compact representation of the car and track context at time `t`
- **Action** `a_t`: steering, acceleration, braking
- **Transition** `P(s_{t+1}|s_t, a_t)`: simulator dynamics
- **Reward** `r_t`: scalar score for this step
- **Discount** `gamma`: how much we value future rewards vs immediate rewards

The policy should maximize:

`E[sum_{t=0..T} gamma^t * r_t]`

---

### 1.2 Why this is hard

Racing is difficult because:

- action space is **continuous** (not small discrete choices)
- dynamics are nonlinear (traction, slip, angle recovery)
- delayed consequences (bad corner entry hurts many future steps)
- sparse milestone events (lap completion) mixed with dense shaping reward
- exploration is dangerous (off-track/spin can terminate episode)

This makes it a strong fit for modern off-policy continuous-control RL methods like SAC.

---

## 2) Why SAC (Soft Actor-Critic)?

### 2.1 Core SAC idea

Standard RL maximizes expected return.
SAC maximizes:

`expected return + alpha * entropy`

Where entropy encourages stochasticity/exploration.
So SAC optimizes both:

- **high reward**
- **sufficient randomness** to avoid brittle local minima

This makes training more stable and sample-efficient than many older methods.

---

### 2.2 SAC components in your code

In `sac.py`:

- **Actor network**: outputs action distribution parameters (`mu`, `log_std`)
- **Twin Critic networks (Q1, Q2)**: estimate action value to reduce overestimation bias
- **Target critic**: slow-moving bootstrap target for stability
- **Temperature parameter** `alpha` (via `log_alpha`): automatically tuned with target entropy
- **Replay buffer**: off-policy learning from stored transitions

---

## 3) TORCS Environment Ground-Up

Your `gym_torcs_sac.py` wraps TORCS so RL code can treat it like a normal environment (`reset()`, `step()`).

### 3.1 Simulator orchestration

The wrapper:

- kills old TORCS processes (`pkill -f torcs`)
- launches TORCS (`TORCS_BIN` override supported)
- triggers auto start script (`autostart.sh`)
- ensures config files are sane (`scr_server`, raceman XML patching)
- forces track selection (`aalborg`, category `road`)

This removes manual GUI setup and makes runs reproducible.

---

### 3.2 What the agent observes

In non-vision mode, observation includes normalized signals like:

- angle to track axis
- track rangefinder array
- lateral position (`trackPos`)
- speed components (`speedX`, `speedY`, `speedZ`)
- wheel spin velocities
- RPM
- fuel-derived features:
  - `fuelLevel`
  - `fuelConsumed`
  - `fuelBudgetRemaining`

`sac.py` then flattens selected fields into a **32-dimensional state vector**.

---

### 3.3 Action space

Agent outputs 3 continuous controls:

- steer in `[-1, 1]`
- accel in `[0, 1]`
- brake in `[0, 1]`

Internally actor samples `tanh`-bounded values in `[-1,1]`.
Then `accel`/`brake` channels are remapped to `[0,1]`.

---

### 3.4 Reward design (very important for presentation)

Reward is not a single metric; it is shaped from multiple terms:

1. **Progress reward**
   - based on forward velocity projection along track direction
   - penalizes lateral/sliding component

2. **Fuel overuse penalty**
   - rolling average fuel burn (20-step window)
   - penalty only for consumption above budget target
   - encourages efficiency without punishing normal driving

3. **Backward motion penalty**
   - negative reward if speedX indicates reverse behavior

4. **Damage penalty**
   - penalizes collision increments

5. **Sector/waypoint bonus**
   - every 500m, rewards faster-than-target pace

6. **Lap completion bonus**
   - strong bonus inversely related to lap time

This is a classic reward-shaping strategy: dense local guidance + sparse global objective.

---

### 3.5 Episode termination logic

Episode can end when:

- car leaves track (`abs(trackPos) > 0.999`)
- spin out (`abs(angle) > 1.0`)
- stuck for too long (`speed < threshold` for >150 steps)
- driving backward (negative alignment)
- lap quota reached (currently 1 lap)

Each failure mode gets a strong terminal penalty.
This creates a clear safety boundary in learning.

---

## 4) SAC Math and Intuition Mapped to Your Code

### 4.1 Actor (policy)

Actor outputs `(mu, log_std)` then samples:

`x_t = mu + std * eps` (reparameterization trick)

`a_t = tanh(x_t)` to bound action.

Log-probability includes tanh correction term:

`log_prob -= log(1 - a^2 + eps)`

This is critical for correct SAC gradients with squashed Gaussian policies.

---

### 4.2 Twin Q critics

Two Q networks estimate `Q1(s,a)` and `Q2(s,a)`.
Training target uses the **minimum** to reduce positive bias:

`target_Q = r + gamma * (min(Q1',Q2') - alpha * log_pi(a'|s'))`

This is central SAC stabilization.

---

### 4.3 Temperature alpha auto-tuning

`alpha` controls exploration-vs-exploitation.
Rather than manually fixing it, your code learns `log_alpha` with target entropy.

If policy is too deterministic, alpha increases (more entropy pressure).
If too random, alpha decreases.

This adaptive mechanism is a key SAC advantage over fixed-noise methods.

---

### 4.4 Replay buffer and off-policy updates

Transitions are stored in replay memory.
During training, random mini-batches break temporal correlation and improve sample efficiency.

Training flow:

1. collect transition
2. store in replay buffer
3. once buffer warm-up threshold is met, sample batch
4. update critics
5. update actor
6. update alpha
7. soft-update target critic

---

### 4.5 Target network soft update

Your update:

`target = tau * online + (1 - tau) * target`

with small `tau = 0.001`.

This low-pass filtering stabilizes bootstrap targets and prevents oscillation.

---

## 5) Architecture Choices in `sac.py`

### 5.1 Network shapes

- Actor: `state -> 300 -> 600 -> (mu, log_std)`
- Critic (each branch): `(state,action) -> 300 -> 600 -> Q`

BatchNorm appears in early layers to help stabilize feature scales.

---

### 5.2 Initialization bias toward moving forward

Actor output bias is initialized to roughly:

- steer ~ 0
- accel ~ high
- brake ~ low

This helps avoid dead starts and gives useful initial exploration behavior for racing.

---

### 5.3 Warm-up strategy

Before replay has enough data, actions are sampled from a biased random distribution:

- small steering noise
- mostly acceleration
- minimal braking

This avoids wasting early experience on obviously non-driving behaviors.

---

## 6) Training Lifecycle in Your Script

### 6.1 Startup and checkpoint behavior

Current behavior (after your recent update):

- default train mode: `--train 1`
- default resume enabled: `--resume 1`
- default checkpoint prefix: `sac_best_model`
- artifact dir default: `./artifacts`

If checkpoint exists and resume is enabled, training continues from it.

---

### 6.2 "Best model" checkpoint policy

Model is saved when episode reward exceeds previous best in this run:

- `sac_best_model_actor.pth`
- `sac_best_model_critic.pth`
- `sac_best_model_critic_target.pth`
- `sac_best_model_log_alpha.pth`

Important nuance: this is best-by-episode-return, not periodic snapshots.

---

### 6.3 Eval mode

Eval mode uses deterministic mean action from actor (`evaluate=True` path).
No training updates are performed.

You can run via:

- `python sac.py --train 0 ...`
- or `./run_saved_agent.sh`

---

## 7) Grounded Walkthrough of `gym_torcs_sac.py`

### 7.1 Why so much config logic?

TORCS can be brittle across installations.
Your wrapper proactively fixes likely breakpoints:

- missing `scr_server.xml`
- race config drift
- malformed multi-driver blocks
- track mismatch

This increases robustness and reproducibility for automated RL loops.

---

### 7.2 Fuel-related features and shaping

You augment state and reward with fuel signals to encourage economically efficient control:

- state includes current normalized fuel and budget projection
- reward penalizes over-budget burn rate using rolling average

This is a useful example of injecting domain constraints into RL.

---

### 7.3 Lap accounting

The code tracks:

- `distFromStart` wrap-around to detect lap crossing
- lap timing from server (`lastLapTime`) fallback to step-derived estimate
- lap logs to `lap_times.log`

This gives clean metrics for presentation: lap completion rate, lap time trends, failure modes.

---

## 8) What to Say in a Presentation (Suggested Story)

### 8.1 Problem framing

"We formulate autonomous racing as continuous-control RL under safety and efficiency constraints."

### 8.2 Method choice

"We use Soft Actor-Critic because it is off-policy, sample-efficient, entropy-regularized, and stable with twin critics."

### 8.3 Environment engineering

"A large portion of practical success is environment reliability: automated TORCS launch/config patching, robust resets, meaningful observations, and clear termination conditions."

### 8.4 Reward design

"Reward blends progress, stability, collision awareness, pace milestones, lap completion, and fuel budget compliance."

### 8.5 Training protocol

"Warm-up exploration -> replay-based updates -> adaptive entropy tuning -> best-checkpoint tracking."

### 8.6 Results framing

Use these metrics:

- episode return over training
- lap completion rate
- best lap time
- off-track/spin/stuck frequency
- fuel budget adherence

---

## 9) Common Audience Questions and Strong Answers

### Q1) Why not DDPG or PPO?

- DDPG can be brittle and sensitive to exploration noise.
- PPO is on-policy and often less sample-efficient for simulators.
- SAC gives a strong stability/sample-efficiency balance for continuous control.

### Q2) Why twin critics?

To reduce overestimation bias from bootstrapping; using `min(Q1,Q2)` is a bias-control mechanism.

### Q3) Why entropy regularization?

It preserves exploration and avoids premature deterministic collapse in complex dynamics.

### Q4) How do you prevent reward hacking?

By combining multiple terms and hard terminations (off-track, spin, backward, stuck), plus lap-based objectives.

### Q5) Is the policy deterministic at test time?

Yes, eval path uses mean action for stable behavior.

---

## 10) Limitations and Future Improvements

1. **Replay buffer is not persisted**
   - resuming model weights does not resume past experience distribution
2. **Single-track specialization**
   - policy may overfit Aalborg
3. **Reward coefficient sensitivity**
   - shaping terms may need retuning for other tracks/objectives
4. **No explicit curriculum/domain randomization**
   - can reduce generalization
5. **No periodic snapshotting**
   - only best model retained by current criterion

Potential upgrades:

- prioritized replay
- n-step returns
- multi-track training with random starts
- structured evaluation suite (seeded scenarios)
- checkpoint every N episodes + best model

---

## 11) Exact File Roles in Your Repo

- `sac.py`
  - SAC networks, training logic, action selection, checkpointing, CLI
- `gym_torcs_sac.py`
  - TORCS process/config management, observation/reward/termination, RL environment API
- `run_saved_agent.sh`
  - convenience eval-only launcher that loads saved checkpoint

---

## 12) Minimal Command Cheat Sheet

Train/resume:

```bash
python sac.py --train 1 --artifact-dir ./artifacts --model-prefix sac_best_model
```

Eval only:

```bash
python sac.py --train 0 --artifact-dir ./artifacts --model-prefix sac_best_model
```

Eval helper script:

```bash
./run_saved_agent.sh
```

Disable resume (fresh training from scratch):

```bash
python sac.py --train 1 --resume 0
```

---

## 13) Executive Summary (30-second version)

This project trains a continuous-control racing policy in TORCS using Soft Actor-Critic.
The environment wrapper provides robust simulator control, state extraction, reward shaping, and safety-driven terminations.
The SAC agent learns a stochastic policy with twin critics and adaptive entropy tuning, then runs deterministic in evaluation.
Checkpointing now supports true resume behavior, enabling long-running iterative improvement cycles.

