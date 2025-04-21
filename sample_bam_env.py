
import gymnasium as gym
import bam_gym_env
from bam_gym_env.transport import ErrorCode, RoslibpyTransport


# First make transport, this allows it communicate with backend server

transport = RoslibpyTransport("test_ns")
# env = gym.make("CartPole-v1", render_mode="human")
# env = gym.make("bam/GridWorld-v0")
# env = gym.make("bam/MNIST", render_mode="human")
env = gym.make("bam/CartPole", transport=transport, render_mode="human")

reset_on_terminate = True

observation, info = env.reset(seed=42)
print(observation)

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Simulated environments always return observations, but sometimes
    # real environments have issues! need to access full info and check if success
    print(info['header'])
    print(f"reward: {reward} terminated: {terminated} truncated: {truncated}")
    if (info['header']['error_code']['value'] != ErrorCode.SUCCESS):
        print("Skipping this step")
        print(info['header']['error_msg'])
        continue

    if (terminated or truncated) and reset_on_terminate:
        print("Calling Reset")
        observation, info = env.reset()
env.close()