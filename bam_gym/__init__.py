from gymnasium.envs.registration import register

# Local

register(
    id="bam/GridWorld-v0",
    entry_point="bam_gym.envs:GridWorldEnv",
)

register(
    id="bam/MNIST",
    entry_point="bam_gym.envs:MNISTEnv",
)

register(
    id="bam/ClassicBandit",
    entry_point="bam_gym.envs:ClassicBandit",
)

register(
    id="bam/ContextBandit",
    entry_point="bam_gym.envs:ContextBandit",
)



# Remote


register(
    id="bam/CartPole",
    entry_point="bam_gym.envs:CartPole",
)

register(
    id="bam/GraspXYR",
    entry_point="bam_gym.envs:GraspXYR",
)
