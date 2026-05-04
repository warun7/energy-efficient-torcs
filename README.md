# Energy-Efficient Autonomous Racing in TORCS (Budget-Conditioned SAC)

This repository demonstrates the training of a **Budget-Conditioned Soft Actor-Critic (SAC)** agent to play TORCS. Moving beyond standard lap-time optimization, this project formulates autonomous racing as an energy management problem (akin to Formula E). 

The agent learns a parameterized policy $\pi(a_t | o_t, b_t)$ that takes the remaining energy budget $b_t$ as input. It automatically adapts its driving style:
- **Unconstrained Energy:** Aggressive, time-attack pacing.
- **Tight Energy Constraints:** "Conservation mode" utilizing lift-and-coast techniques to maximize progress.

For a deep dive into the RL formulation, reward shaping, and evaluation results, please read the full report: [budget_conditioned_rl_report.md](budget_conditioned_rl_report.md).

## Demo

<video src="rl_demo.mp4" controls width="800"></video>

*(If the video above does not render, you can [download/view rl_demo.mp4 directly](rl_demo.mp4))*

---

## Installation Dependencies:

* Python 3.9+
* TensorFlow (CPU) + `tf_keras` (Keras 2 API on TF 2.x)
* PyTorch (for the SAC implementation)
* NumPy, Matplotlib
* Gymnasium
* TORCS built from `gym_torcs/vtorcs-RL-color` (see `run.sh`)

## Automated run (Ubuntu 22.04 / Docker)

From the repository root (only supported path for course evaluation):

```bash
chmod +x run.sh
./run.sh
```

This installs system packages, builds TORCS with `scr_server` into `./build/torcs-install`, creates `./.venv`, runs a short training phase then evaluation, and writes logs, weights, and plots under `./artifacts/`.

Optional environment variables: `TRAIN_EPISODES`, `EVAL_EPISODES`, `MAX_STEPS`, `APT_UPDATE=0` (skip `apt-get update`).

## Quick Setup (Linux)

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) TORCS: the stock `torcs` package from apt does **not** include the `scr_server` AI driver used by this project. Prefer `./run.sh`, which compiles vtorcs from `gym_torcs/vtorcs-RL-color` into `./build/torcs-install`.

3) Ensure your working directory is this repository root.

## How to Run

### Training the Budget-Conditioned SAC Agent
To train the agent with the randomized budget curriculum (40% tight, 40% mid, 20% loose):
```bash
python sac.py
```
*Note: This script will automatically use the 32-dim unconstrained baseline weights to warm-start the 33-dim policy if `artifacts/baseline_model` is present.*

### Evaluating the Agent
To run deterministic evaluation across various hold-out budgets and see how the agent adapts its pace:
```bash
python eval_sac.py --mode conditioned
```
To run the unconstrained 32-dim baseline model for comparison:
```bash
python eval_sac.py --mode baseline
```

**Notes:**
- `autostart.sh` is used by `gym_torcs_sac.py` to auto-join a race.
- If TORCS does not launch, verify `torcs` is in your `PATH` and that X11/display access is available.
