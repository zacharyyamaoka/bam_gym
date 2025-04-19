from gymnasium.envs.registration import register

register(
    id="bam/GridWorld-v0",
    entry_point="bam_gym_env.envs:GridWorldEnv",
)

register(
    id="bam/MNIST",
    entry_point="bam_gym_env.envs:MNISTEnv",
)

register(
    id="bam/CartPole",
    entry_point="bam_gym_env.envs:CartPole",
)
