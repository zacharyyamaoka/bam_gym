from gymnasium.envs.registration import register

register(
    id="bam/GridWorld-v0",
    entry_point="bam_gym.envs:GridWorldEnv",
)

register(
    id="bam/MNIST",
    entry_point="bam_gym.envs:MNISTEnv",
)

register(
    id="bam/CartPole",
    entry_point="bam_gym.envs:CartPole",
)

register(
    id="bam/GraspV1",
    entry_point="bam_gym.envs:GraspV1",
)
