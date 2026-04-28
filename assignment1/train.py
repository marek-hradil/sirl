from stable_baselines3 import PPO
from main import make_backwards_environment, seed, SEED, LEARNING_TIMESTEPS


def train_model(env):
    model = PPO("MlpPolicy", env, seed=SEED, verbose=1)
    model.learn(total_timesteps=LEARNING_TIMESTEPS)

    env.reset()

    return model


def main():
    seed()
    env = make_backwards_environment()
    model = train_model(env)

    model.save("cheetah_forward")

    env.close()


main()
