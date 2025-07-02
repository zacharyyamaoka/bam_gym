#!/usr/bin/env python3

import pytest

# BAM
import bam_gym
from bam_gym.wrappers import MockEnv, MockObs
from bam_gym.policies.generic_policy import GenericPolicy
from bam_gym.utils import print_step
# PYTHON
import numpy as np
import gymnasium as gym

""" Design Notes:
    
    This is really the heart of the entire system. some type of policy itneracting with an MDP
    - Making your own bandit mdps were you define a ground truth distrbution and try to learn it 
    can be great I think...
    Simpler script that provides a util to just run an agent for a certain number of steps..

    Mabye make it yield? or iterate so you can get the result each time?
"""

# for result in run_agent

def run_agent(policy: GenericPolicy,
              env: gym.Env,
              n_steps=10):
    
    policy.env_init(env)
    obs, info = env.reset()
    reward, terminated, truncated = None, None, None

    for iter in range(n_steps):

        action, action_info = policy(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        print_step(action, obs, reward, terminated, truncated, info, iter)
        if terminated or truncated:
            obs, info = env.reset()

def main():
    env = MockObs(gym.make('bam/GenericGymClient', disable_env_checker=True, n_pose=1))
    # policy = bam_gym.make_policy('BlindPolicy')
    policy = bam_gym.make_policy('RandomPolicy')

    run_agent(policy, env, n_steps=10)

if __name__ == "__main__":
    main()