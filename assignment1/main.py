import gymnasium as gym
from gymnasium import RewardWrapper
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np
import random

SEED = 1
LEARNING_TIMESTEPS = 2_000_000


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)  # optional, strictest mode


class UprightReward(RewardWrapper):
    def reward(self, reward):
        torso_height = self.unwrapped.data.body("torso").xpos[2]
        return reward - 2.0 * max(0.0, 0.45 - torso_height)


class BackwardsReward(RewardWrapper):
    def reward(self, reward):
        return -reward


def make_forwards_environment(render=False):
    env = Monitor(
        UprightReward(
            gym.make("HalfCheetah-v4", render_mode="human" if render else None)
        ),
        filename="./logs.csv",
    )

    env.action_space.seed(SEED)
    return env


def make_backwards_environment(render=False):
    env = Monitor(
        BackwardsReward(
            gym.make("HalfCheetah-v4", render_mode="human" if render else None)
        ),
        filename="./logs.csv",
    )

    env.action_space.seed(SEED)
    return env
