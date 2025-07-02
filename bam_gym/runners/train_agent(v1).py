#!/usr/bin/env python3

import pytest

# BAM
import bam_gym
from bam_gym.wrappers import MockEnv, MockObs
from bam_gym.policies.generic_policy import GenericPolicy

# PYTHON
import numpy as np
import gymnasium as gym
import argparse

""" Design Notes:

    Heart of the system.

    Playing a gym env in an iterative manner.

    Reference berkley training code? I htink there is a light verison and a train_agent() version?

    Don't pass the class type in, then you need to pass in all the params as well! better to pass in an instance

    See script from: 
        https://github.com/berkeleydeeprlcourse/homework_fall2023/blob/main/hw1/cs285/scripts/run_hw1.py


    Using args is cool beacuse then you can save the scripts you used to run the experiment/save to yaml. 
    Its very similar to ros2 launch!

    ```bash
    python cs285/scripts/run_hw1.py \
        --expert_policy_file cs285/policies/experts/Ant.pkl \
        --env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
        --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
        --video_log_freq -1
    ```
"""



def run_agent(policy: GenericPolicy,
              env: gym.Env,
              n_steps=10):

def main():
    env = MockObs(gym.make('bam/GenericGymClient', disable_env_checker=True, n_pose=1))
    policy = bam_gym.make_policy('BlindPolicy')

    #region - Load in args
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)
    #endregion - Load in args

    # #region - Process/validate args
    # if args.do_dagger:
    #     # Use this prefix when submitting. The auto-grader uses this prefix.
    #     logdir_prefix = 'q2_'
    #     assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    # else:
    #     # Use this prefix when submitting. The auto-grader uses this prefix.
    #     logdir_prefix = 'q1_'
    #     assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # # directory for logging
    # data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    # if not (os.path.exists(data_path)):
    #     os.makedirs(data_path)
    # logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    # logdir = os.path.join(data_path, logdir)
    # params['logdir'] = logdir
    # if not(os.path.exists(logdir)):
    #     os.makedirs(logdir)

    #endregion - Process/validate args




if __name__ == "__main__":
    main()