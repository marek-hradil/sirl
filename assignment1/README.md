# Assignment 1 — HalfCheetah PPO

Training and observing a PPO agent on the MuJoCo `HalfCheetah-v4` environment using Stable-Baselines3.

## Files

- `main.py` — shared config (seed, timesteps) and environment factories with reward wrappers
- `train.py` — trains a PPO agent and saves the model to `cheetah_forward.zip`
- `observe.py` — loads a saved model and renders one episode with a tracking camera
- `cheetah_modified.xml` — modified MuJoCo XML for the HalfCheetah model
- `logs.csv.monitor.csv` — episode reward/length log written by the Monitor wrapper

## Reward wrappers

- `UprightReward` — penalises the agent when the torso drops below height 0.45, discouraging flipping
- `BackwardsReward` — negates the forward velocity reward to train a backwards-running agent

## Usage

```bash
# Train
uv run train.py

# Observe
uv run observe.py
```

## Config (`main.py`)

| Variable | Default | Description |
|---|---|---|
| `SEED` | `1` | Global random seed |
| `LEARNING_TIMESTEPS` | `2_000_000` | Total training steps |
| `PLAYBACK_SPEED` (`observe.py`) | `0.75` | Render speed (1.0 = real-time) |
