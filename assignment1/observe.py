import time
import mujoco
from stable_baselines3 import PPO
from main import make_backwards_environment, make_forwards_environment, seed

PLAYBACK_SPEED = 0.75  # 1.0 = real-time, 0.5 = half speed


def setup_tracking_camera(env):
    base_env = env.unwrapped
    viewer = base_env.mujoco_renderer.viewer
    if viewer is not None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_env.model.body("torso").id


def observe_trajectory(env, model):
    total_reward: float = 0.0
    episode_over = False

    observation, _ = env.reset()
    setup_tracking_camera(env)

    dt = env.unwrapped.dt
    while not episode_over:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        episode_over = terminated or truncated
        time.sleep(dt / PLAYBACK_SPEED)

    print(f"Episode finished! Total reward: {total_reward}")


def main():
    seed()
    env = make_backwards_environment(render=True)

    model = PPO.load("cheetah_forward")
    observe_trajectory(env, model)

    env.close()


main()
