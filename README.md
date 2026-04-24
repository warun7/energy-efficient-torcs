## Using Keras and Deep Deterministic Policy Gradient to play TORCS

300 lines of python code to demonstrate DDPG with Keras

Please read the following blog for details

https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

![](fast.gif)

# Installation Dependencies (updated):

* Python 3.9+
* TensorFlow (CPU) + `tf_keras` (Keras 2 API on TF 2.x)
* NumPy, Matplotlib
* Gymnasium
* TORCS built from `gym_torcs/vtorcs-RL-color` (see `run.sh`)

# Automated run (Ubuntu 22.04 / Docker)

From the repository root (only supported path for course evaluation):

```bash
chmod +x run.sh
./run.sh
```

This installs system packages, builds TORCS with `scr_server` into `./build/torcs-install`, creates `./.venv`, runs a short training phase then evaluation, and writes logs, weights, and plots under `./artifacts/`.

Optional environment variables: `TRAIN_EPISODES`, `EVAL_EPISODES`, `MAX_STEPS`, `APT_UPDATE=0` (skip `apt-get update`).

# Quick Setup (Linux)

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) TORCS: the stock `torcs` package from apt does **not** include the `scr_server` AI driver used by this project. Prefer `./run.sh`, which compiles vtorcs from `gym_torcs/vtorcs-RL-color` into `./build/torcs-install`.

3) Ensure your working directory is this repository root (where `ddpg.py` exists), then run:

```bash
python ddpg.py
```

This starts inference mode by default (`train_indicator=0` in `playGame`).

To train, run:

```bash
python ddpg.py --train 1 --artifact-dir ./artifacts --episodes 2000
```

Notes:
- `autostart.sh` is used by `gym_torcs.py` to auto-join a race.
- If TORCS does not launch, verify `torcs` is in your `PATH` and that X11/display access is available.
