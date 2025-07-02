import time

"""
What I learned. 

Even if you directly load a module it will call the __init__.py file.

This means that if you have a __init__.py file that imports a lot of stuff, it will take a long time to import.

This is why we need to lazy load the modules.

It seems there is some basic overhead to the first import of about 150ms. so the first import may seem a bit slower

Ok I should not just go trigger happy and import everything always into the __init__.py file, just when it would be actually useful!

"""
start = time.time()
from bam_gym.envs.local_server.context_bandit import ContextBandit
print(f"ContextBandit Import took {time.time() - start:.2f} seconds")

start = time.time()
from bam_gym.wrappers.mock_env import MockEnv
print(f"MockEnv Import took {time.time() - start:.2f} seconds")

start = time.time()
from bam_gym.envs.local_server.grasp_net_env import GraspNetEnv
print(f"GraspNetEnvImport took {time.time() - start:.2f} seconds")
