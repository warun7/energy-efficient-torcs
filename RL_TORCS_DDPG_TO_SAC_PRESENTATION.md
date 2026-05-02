# RL in TORCS: From DDPG to SAC (First Principles + Project Evolution)

This document is a presentation-ready, ground-up explanation of:

1. The autonomous racing RL problem
2. The first approach: `ddpg.py` + `gym_torcs/gym_torcs.py`
3. How DDPG works mathematically and in your implementation
4. Observed shortcomings and why they matter
5. The shift to SAC: `sac.py` + `gym_torcs_sac.py`
6. A precise side-by-side comparison of RL components in both approaches

---

## 1) Problem Definition From Ground Zero

### 1.1 Task

Train an agent to drive a car in TORCS autonomously and efficiently.
The agent should stay on track, make progress quickly, avoid damage, and finish laps.

---

### 1.2 MDP Formulation

The problem is modeled as a **Markov Decision Process (MDP)**:

- **State space** `S`: all sensory information the agent can observe
- **Action space** `A`: continuous control outputs (steering/throttle/brake)
- **Transition dynamics** `P(s'|s,a)`: TORCS physics/simulator response
- **Reward function** `R(s,a,s')`: scalar feedback signal
- **Discount factor** `gamma`: tradeoff between immediate and future returns
- **Policy** `pi(a|s)` or deterministic `mu(s)`: control law learned by RL

Objective:

`max E[sum_t gamma^t r_t]`

---

### 1.3 Why this is hard

- Continuous actions
- Nonlinear dynamics (slip, heading correction, momentum)
- High sensitivity to exploration quality
- Delayed consequences from earlier control errors
- Need for safety and stability during exploration

---

## 2) Approach 1: DDPG + Original TORCS Wrapper

Files:

- `ddpg.py`
- `gym_torcs/gym_torcs.py`

---

## 3) DDPG Fundamentals (First Principles)

### 3.1 What DDPG is

DDPG (Deep Deterministic Policy Gradient) is an **off-policy actor-critic** algorithm for continuous control.

It has:

- **Actor**: deterministic policy `a = mu(s)`
- **Critic**: value estimator `Q(s,a)`
- **Target networks** for both actor and critic
- **Replay buffer** for sample reuse and decorrelation

---

### 3.2 Deterministic policy gradient intuition

Instead of sampling actions stochastically, actor outputs a single action.
Actor is trained to choose actions that maximize critic output:

`nabla_theta J ~ E[nabla_a Q(s,a)|_{a=mu(s)} * nabla_theta mu(s)]`

So:

- critic learns "how good this action is"
- actor learns to move actions toward higher-Q regions

---

### 3.3 Critic learning

Critic learns Bellman target:

`y = r + gamma * Q_target(s', mu_target(s'))` (if not terminal)

Loss:

`L = (Q(s,a) - y)^2`

---

### 3.4 Exploration in DDPG

Because policy is deterministic, exploration is added as external action noise.
Your implementation uses **Ornstein-Uhlenbeck (OU) noise** with decaying epsilon.

This is a key practical detail and also one source of fragility.

---

## 4) DDPG Setup in Your Actual Code

### 4.1 RL components in `ddpg.py`

- **State vector dimension**: `29`
- **Action dimension**: `3` (`steer`, `accel`, `brake`)
- **Replay buffer size**: `100000`
- **Batch size**: `32`
- **Gamma**: `0.99`
- **Tau**: `0.001`
- **Actor LR**: `1e-4`
- **Critic LR**: `1e-3`

State construction:

`[angle, track, trackPos, speedX, speedY, speedZ, wheelSpinVel/100, rpm]`

---

### 4.2 Training loop behavior

Per step:

1. Actor predicts action
2. OU noise added (scaled by decaying epsilon)
3. Step env, collect `(s,a,r,s',done)`
4. Store in replay
5. Sample minibatch
6. Compute critic target
7. Update critic
8. Compute action gradients from critic
9. Update actor
10. Soft-update target networks

Weights are saved every episode to:

- `actormodel.h5`
- `criticmodel.h5`

---

## 5) Original TORCS Env (`gym_torcs/gym_torcs.py`) in RL Terms

### 5.1 State space definition (original)

Observation object contains (non-vision mode):

- focus
- speedX, speedY, speedZ
- opponents
- rpm
- track
- wheelSpinVel

But training uses a flattened 29-d state with:

- angle
- track
- trackPos
- speeds
- wheelSpinVel
- rpm

---

### 5.2 Action space definition (original)

For throttle-enabled mode:

- steer from action[0]
- accel from action[1]
- brake is effectively controlled in policy dimension but env mapping in this file is simpler than SAC wrapper

Also gear is auto-handled in wrapper unless manual gear mode enabled.

---

### 5.3 Reward formulation (original)

Primary reward:

- `reward = speedX * cos(angle)` (forward progress component)

Penalties:

- collision damage increase => reward `-1`
- out of track => reward `-1` + terminate

Terminations:

- out of track
- too little progress after warmup window
- running backward (`cos(angle) < 0`)

---

## 6) How DDPG Performed and Why It Hit Limits

Use this section in your talk as "what worked vs what broke."

### 6.1 What worked

- Learned basic forward-driving behavior
- Off-policy replay improved sample reuse
- Continuous control was feasible in TORCS
- Could get initial policy learning with dense progress reward

### 6.2 Core shortcomings observed/expected for this setup

1. **Deterministic policy brittleness**
   - exploration quality depends heavily on OU noise tuning
   - small noise changes can destabilize behavior

2. **Q overestimation risk**
   - single critic in vanilla DDPG prone to optimistic bias
   - can push actor toward bad action regions

3. **Exploration decay issue**
   - epsilon-decayed OU noise can become too weak too early
   - policy may converge prematurely to suboptimal behavior

4. **Simpler reward shaping**
   - strong reliance on progress term with limited structure
   - can learn locally good but globally weak racing behavior

5. **Environment/reliability constraints**
   - older wrapper has less robust startup/config management than SAC wrapper
   - reproducibility and stability become practical bottlenecks

6. **Checkpoint limitations in practice**
   - saving model weights only (no richer training state)
   - long-run continuation quality depends on many external factors

---

## 7) Why Move to SAC

SAC was chosen because it directly addresses DDPG pain points:

- stochastic policy with entropy regularization => stronger, adaptive exploration
- twin critics => reduced overestimation bias
- temperature tuning (`alpha`) => automatic exploration/exploitation balancing
- typically better stability and robustness in continuous control

In short: SAC gives more reliable optimization behavior under noisy, nonlinear racing dynamics.

---

## 8) SAC Fundamentals (for contrast)

### 8.1 Objective

SAC maximizes:

`E[sum gamma^t (r_t + alpha * H(pi(.|s_t)))]`

Entropy term keeps policy sufficiently exploratory.

### 8.2 Key components

- stochastic Gaussian actor (with tanh squashing)
- twin critics `Q1`, `Q2`
- target critic
- replay buffer
- automatic alpha tuning using target entropy

---

## 9) SAC Setup in Your Project

Files:

- `sac.py`
- `gym_torcs_sac.py`

Notable improvements in env/reward engineering:

- richer state features (fuel-level, fuel-consumed, fuel-budget signals)
- richer reward (progress + damage + overconsumption + waypoint + lap bonus)
- clearer high-penalty terminal conditions (off-track/spin/stuck/backward)
- lap tracking/logging support
- more robust TORCS config normalization and launch handling

---

## 10) RL Component Definitions: DDPG vs SAC (Side-by-Side)

### 10.1 State space

**DDPG (`ddpg.py` + `gym_torcs/gym_torcs.py`)**
- flattened 29-D vector
- mostly geometric/kinematic signals (angle, track, trackPos, speed, wheel, rpm)

**SAC (`sac.py` + `gym_torcs_sac.py`)**
- flattened 32-D vector
- includes all core driving features plus fuel-related signals for constrained behavior

---

### 10.2 Action space

**DDPG**
- 3 continuous outputs for steer/accel/brake (with deterministic actor + external OU noise)

**SAC**
- 3 continuous outputs from squashed Gaussian policy
- steer in `[-1,1]`, accel/brake remapped to `[0,1]`

---

### 10.3 Reward formulation

**DDPG env**
- mainly forward progress (`speedX * cos(angle)`)
- simple collision/out-of-track penalties

**SAC env**
- progress with slip awareness
- damage penalties
- fuel budget overuse penalty
- waypoint pace bonuses
- lap completion bonus
- stronger terminal penalties

---

### 10.4 Policy type

**DDPG**
- deterministic policy `mu(s)`
- exploration injected externally by OU noise

**SAC**
- stochastic policy `pi(a|s)` with learned variance
- entropy objective encourages intrinsic exploration

---

### 10.5 Critic design

**DDPG**
- single critic target estimate

**SAC**
- twin critics, target uses `min(Q1,Q2)` for bias control

---

### 10.6 Stability tools

**DDPG**
- replay buffer + target networks
- OU noise schedule sensitive

**SAC**
- replay buffer + target networks
- entropy regularization
- adaptive alpha
- twin-critic bias mitigation

---

## 11) Presentation Narrative You Can Use

### Slide flow suggestion

1. **Problem framing** (autonomous racing as MDP)
2. **Initial baseline: DDPG**
3. **DDPG algorithm from first principles**
4. **How DDPG was implemented in our code**
5. **Observed weaknesses / instability causes**
6. **Why SAC is a better fit**
7. **SAC algorithm from first principles**
8. **Environment/reward redesign for SAC**
9. **DDPG vs SAC component-by-component**
10. **Takeaways + future work**

---

## 12) Strong Q&A Answers

### Q: Why didn’t DDPG fully solve it?

DDPG worked as a useful baseline, but deterministic policy + external noise made exploration brittle, and the single-critic setup is more vulnerable to value overestimation and unstable improvements in this environment.

### Q: Why is SAC better here?

SAC gives robust exploration through entropy regularization, controls overestimation via twin critics, and adapts exploration pressure automatically via alpha tuning.

### Q: Was the improvement only algorithmic?

No. Improvement came from both algorithm choice and environment/reward engineering upgrades (richer signals, more structured shaping, safer terminations, better simulator reliability logic).

---

## 13) Honest Limitations (Both Approaches)

- replay buffers are not persisted across runs
- potential single-track overfitting
- reward-shaping coefficients may need retuning for other tracks
- no full domain randomization/curriculum yet
- benchmark protocol can be strengthened with seeded eval suites

---

## 14) One-Minute Executive Summary

We started with DDPG as a baseline continuous-control actor-critic in TORCS.
It proved feasibility but exposed practical limitations: brittle exploration, stability sensitivity, and limited robustness.
We then moved to SAC, which adds entropy-regularized stochastic policies, twin critics, and adaptive temperature tuning.
Combined with a stronger environment and reward formulation, SAC provides a more stable and scalable approach for autonomous racing in this project.

