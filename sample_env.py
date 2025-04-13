import gymnasium as gym
import bam_gym_env
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("bam/GridWorld-v0", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()