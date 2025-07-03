#!/usr/bin/env python3

import pytest

# BAM

import bam_gym


# PYTHON
import numpy as np
import gymnasium as gym

def test_passive_env_checker():
    # for env_spec in gym.envs.registry.values():
    #     if "bam/" in env_spec.id:
    #         # print(env_spec.id)
    #         pass

    env = gym.make('bam/GenericGymClient', disable_env_checker=False)
    env.reset()
    for i in range(3):
        env.step(env.action_space.sample())
    env.close()

    env = gym.make('bam/PickClient', disable_env_checker=False)
    env.reset()
    print(env.action_space)
    for i in range(3):
        env.step(env.action_space.sample())
    env.close()
