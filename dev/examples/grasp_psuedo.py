#!/usr/bin/env python3

import gymnasium as gym
import bam_gym # you need to import to register
env = gym.make("bam/GraspXY", num_circles=1, render_mode="human")

model = GraspModel()

observation, info = env.reset(seed=42)
# observation['color'], observation['depth'], observation['target_class'], observation['seg']

for _ in range(100):

    # Filter from Millons of options to just 1-8

    # Return a number of (partial) grasps (x,y,z,rx,ry,rz,w) with probability of success
    grasp_heat_map = model(observation['color'], observation['depth'])

    # Filter out grasps that would be on the wrong objects  
    grasp_heat_map = class_filter(grasp_heat_map, observation['target_class'], observation['seg'])

    # You may need to autocomplete the grasp

    # Select grasps following simple quadrant rule for 1,2,4 arms. Priortize perhaps items near edge, etc.
    grasp_list = quadrant_filter(grasp_heat_map, n_arms=1, e_greedy=False, retries=True)

    # ^^ This will change alot depending on approaches

    msg_factory.pick(pick)
    msg_factory.pick_and_place(id, pick, place)
    msg_factory.pick_and_place(id, pick, place, retry_pick)
    msg_factory.pick_and_place(id, pick, place, pick_approach, etc..)

    # Don't worry about msg factories, etc. in here....
    # Positino matters... it encodes the prefix of which arm to do what...
    action = [pick, pick, pick, pick], [pick_retry, pick_retry, pick_retry, pick_retry]

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"Action: {action}")
    print(f"Observation: (shape={observation.shape}, dtype={observation.dtype})")
    # print(f"Observation: {observation}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    if (terminated or truncated):
        observation, info = env.reset()
env.close()

