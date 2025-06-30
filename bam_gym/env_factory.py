import bam_gym

def make_env(name, *args, **kwargs):
    # Load in a lazy way to avoid importing in potetailly large envs
    # from bam_gym.envs.clients import (
    # CartPole,
    # GraspXYR,
    # MockEnv,
    # PickAndLift,
    # ObsEnv,
    # )

    import bam_gym.envs.local_server as local_server
    # TODO likely change this to use passive env loading?

    """
    Factory function to create a client instance by name.
    Args:
        name (str): Name of the client. Supported: 'CartPole', 'GraspXYR', 'MockEnv', 'PickAndLift', 'ObsEnv'
        *args, **kwargs: Passed to the client constructor.
    Returns:
        An instance of the requested client.
    Raises:
        ValueError: If the client name is not recognized.
    """
    if name == "CartPole":
        return bam_gym.envs.local_server.clients.CartPole(*args, **kwargs)
    elif name == "GraspXYR":
        return clients.GraspXYR(*args, **kwargs)
    elif name == "MockEnv":
        return clients.MockEnv(*args, **kwargs)
    elif name == "PickAndLift":
        return clients.PickAndLift(*args, **kwargs)
    elif name == "ObsEnv":
        return clients.ObsEnv(*args, **kwargs)
    else:
        raise ValueError(f"Unknown client name: {name}")
        raise ValueError(f"Unknown client name: {name}")
