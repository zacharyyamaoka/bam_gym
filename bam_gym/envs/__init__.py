from bam_gym.envs.local.grid_world import GridWorldEnv
from bam_gym.envs.local.mnist import MNISTEnv
from bam_gym.envs.local.classic_bandit import ClassicBandit
from bam_gym.envs.local.context_bandit import ContextBandit
# from bam_gym.envs.local.simple_xy import GraspXY

from bam_gym.envs.remote.bam_env import BamEnv # parent goes first
from bam_gym.envs.remote.cart_pole import CartPole
from bam_gym.envs.remote.grasp_v1 import GraspV1
