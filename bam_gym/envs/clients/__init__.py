# Don't do imports here, as it can be expensive, directly load each env as needed
# For cleaner code you can use the env_factory.py

from bam_gym.envs.clients.generic_gym_client import GenericGymClient
# from bam_gym.envs.clients.cart_pole import CartPole
# from bam_gym.envs.clients.grasp_xyr import GraspXYR
# from bam_gym.envs.clients.pick_and_lift import PickAndLift
# from bam_gym.envs.clients.obs_env import ObsEnv
# from bam_gym.envs.clients.mock_env import MockEnv


