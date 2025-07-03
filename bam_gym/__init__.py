from gymnasium.envs.registration import register

# import main base classes as they are used often!
from bam_gym.policies.generic_policy import GenericPolicy
from bam_gym.envs.clients.generic_gym_client import GenericGymClient

from bam_gym.env_factory import make_env
from bam_gym.policy_factory import make_policy

from bam_gym.utils.pprint import print_action, print_observation, print_reset, print_step, print_gym_space

"""
I was getting frusterated as it was taking a while to run the code. I checked the import and it seems I have some large imports
that were taking 3s to load!

Import took 0.30 seconds
WARNING:root:Failed to import ros dependencies in rigid_transforms.py
WARNING:root:autolab_core not installed as catkin package, RigidTransform ros methods will be unavailable
Import took 2.75 seconds
Import took 0.00 seconds

I think this is why gym uses the register pattern, so avoid having to load all the envs at once

Ok there is actually alot of nice utilities that gymnasium provides. and structure, that would be good to follow to get a big speed up on API help..

https://gymnasium.farama.org/api/vector/utils/

It's a balance certaintly... though 
"""
# Local

register(
    id="bam/GridWorld-v0",
    entry_point="bam_gym.envs.local_server.grid_world:GridWorldEnv",
)

register(
    id="bam/MNIST",
    entry_point="bam_gym.envs.local_server.mnist:MNISTEnv",
)

register(
    id="bam/ClassicBandit",
    entry_point="bam_gym.envs.local_server.classic_bandit:ClassicBandit",
)

register(
    id="bam/ContextBandit",
    entry_point="bam_gym.envs.local_server.context_bandit:ContextBandit",
)



# Remote


register(
    id="bam/CartPole",
    entry_point="bam_gym.envs.clients.cart_pole:CartPole",
)

register(
    id="bam/GenericGymClient",
    entry_point="bam_gym.envs.clients.generic_gym_client:GenericGymClient",
)

register(
    id="bam/PickClient",
    entry_point="bam_gym.envs.clients.pick_client:PickClient",
)

