#!/usr/bin/env python3

"""
Example demonstrating how to use the BAM Vector Environment

This example shows how to create and use a vectorized environment
that maintains a single client in the background while providing
vector environment functionality.
"""

import numpy as np
from bam_gym.envs import make_bam_vec_env
from bam_gym.envs.clients.mock_env import MockEnv
from bam_gym.transport import MockTransport


def create_mock_env(**kwargs):
    """Factory function to create a mock environment."""
    transport = MockTransport()
    return MockEnv(transport=transport, **kwargs)


def main():
    """Demonstrate vector environment usage."""
    
    # Create a vectorized environment with 4 parallel environments
    num_envs = 4
    vec_env = make_bam_vec_env(
        env_factory=create_mock_env,
        num_envs=num_envs,
        obs=True,  # Enable observations
        color=True,  # Enable color images
        render_mode="rgb_array"  # Enable rendering
    )
    
    print(f"Vector Environment created with {vec_env.num_envs} environments")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    print(f"Single observation space: {vec_env.single_observation_space}")
    print(f"Single action space: {vec_env.single_action_space}")
    
    # Reset all environments
    observations, infos = vec_env.reset(seed=42)
    print(f"\nReset completed:")
    print(f"Observations type: {type(observations)}")
    print(f"Number of observations: {len(observations)}")
    print(f"Info keys: {list(infos.keys()) if infos else 'None'}")
    
    # Take actions in all environments
    actions = vec_env.action_space.sample()
    print(f"\nSampled actions: {actions}")
    print(f"Actions shape: {actions.shape if hasattr(actions, 'shape') else 'N/A'}")
    
    # Step all environments
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    
    print(f"\nStep completed:")
    print(f"Rewards: {rewards}")
    print(f"Terminations: {terminations}")
    print(f"Truncations: {truncations}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Terminations shape: {terminations.shape}")
    
    # Take a few more steps
    for step in range(3):
        actions = vec_env.action_space.sample()
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        print(f"Step {step + 2}: Rewards = {rewards}, Terminations = {terminations}")
        
        # Check if any environment terminated
        if np.any(terminations):
            print(f"Some environments terminated at step {step + 2}")
    
    # Render (if supported)
    try:
        render_output = vec_env.render()
        if render_output is not None:
            print(f"\nRender output shape: {render_output.shape if hasattr(render_output, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"Rendering failed: {e}")
    
    # Close the environment
    vec_env.close()
    print("\nEnvironment closed successfully")


if __name__ == "__main__":
    main() 