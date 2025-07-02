#!/usr/bin/env python3

"""
Follow gym registration pattern to allow lazy loading of policies
"""
from bam_gym.policies.generic_policy import GenericPolicy

def make_policy(name, *args, **kwargs) -> GenericPolicy:
    """
    Factory function to create an agent instance by name.
    Args:
        name (str): Name of the agent. Supported: 'CartPoleAgent', 'GraspXYRAgent', 'MockAgent', 'PickAndLiftAgent', 'ObsAgent'
        *args, **kwargs: Passed to the agent constructor.
    Returns:
        An instance of the requested agent.
    Raises:
        ValueError: If the agent name is not recognized.
    """
    if name == "RandomPolicy":
        from bam_gym.policies.random_policy import RandomPolicy
        return RandomPolicy(*args, **kwargs)
    elif name == "BlindPolicy":
        from bam_gym.policies.blind_policy import BlindPolicy
        return BlindPolicy(*args, **kwargs)
    # elif name == "MockAgent":
    #     return MockAgent(*args, **kwargs)
    # elif name == "PickAndLiftAgent":
    #     return PickAndLiftAgent(*args, **kwargs)
    # elif name == "ObsAgent":
    #     return ObsAgent(*args, **kwargs)
    else:
        raise ValueError(f"Unknown agent name: {name}")