# BAM Vector Environment Guide

## Overview

The BAM Vector Environment provides vectorization support for BAM Gym environments while maintaining a single client in the background. This is different from standard Gymnasium vector environments because all vector operations go through the same GymAPI transport, ensuring consistency and efficiency.

## Key Features

- **Single Client Architecture**: Only one actual client/transport is used, reducing resource usage
- **Shared GymAPI Connection**: All environments share the same underlying GymAPI connection
- **Automatic Batching**: Handles batching and unbatching of observations, actions, rewards, etc.
- **Gymnasium Compatible**: Implements the standard Gymnasium VectorEnv interface

## Usage

### Basic Usage

```python
from bam_gym.envs import make_bam_vec_env
from bam_gym.envs.clients.mock_env import MockEnv
from bam_gym.transport import MockTransport

def create_mock_env(**kwargs):
    """Factory function to create a single environment instance."""
    transport = MockTransport()
    return MockEnv(transport=transport, **kwargs)

# Create a vectorized environment with 4 parallel environments
vec_env = make_bam_vec_env(
    env_factory=create_mock_env,
    num_envs=4,
    obs=True,
    color=True,
    render_mode="rgb_array"
)
```

### Environment Operations

```python
# Reset all environments
observations, infos = vec_env.reset(seed=42)

# Take actions in all environments
actions = vec_env.action_space.sample()
observations, rewards, terminations, truncations, infos = vec_env.step(actions)

# Close the environment
vec_env.close()
```

## Architecture

### GenericGymClient Modifications

The `GenericGymClient` class has been enhanced with vector environment support:

- **`num_envs`**: Number of parallel environments
- **`single_observation_space`**: Observation space for a single environment
- **`single_action_space`**: Action space for a single environment
- **`set_action_space()`**: Method to set action spaces for both single and vector modes

### BamVectorEnv Wrapper

The `BamVectorEnv` class provides the vector environment functionality:

- **Single Environment**: Maintains one `GenericGymClient` instance
- **Batched Spaces**: Creates appropriate batched observation and action spaces
- **State Management**: Handles batching/unbatching of all environment data
- **Autoreset Support**: Implements next-step autoreset mode

## Space Handling

### Observation Spaces

The vector environment automatically creates batched observation spaces:

- **Sequence Spaces**: Wrapped in another sequence
- **Dict Spaces**: Each component is batched appropriately
- **Box Spaces**: Batch dimension added with proper low/high bounds

### Action Spaces

Action spaces are batched based on their type:

- **Discrete**: Converted to `MultiDiscrete`
- **Box**: Batch dimension added
- **MultiDiscrete**: Repeated for all environments
- **Other**: Wrapped in sequence

## Integration with Training Libraries

The vector environment is compatible with popular RL training libraries:

### Stable Baselines3

```python
from stable_baselines3 import PPO
from bam_gym.envs import make_bam_vec_env

vec_env = make_bam_vec_env(env_factory, num_envs=4)
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)
```

### RLlib

```python
import ray
from ray import tune
from bam_gym.envs import make_bam_vec_env

ray.init()

def env_creator(env_config):
    return make_bam_vec_env(env_factory, num_envs=4, **env_config)

tune.run(
    "PPO",
    config={
        "env": env_creator,
        "num_workers": 1,
        "num_envs_per_worker": 1,
    }
)
```

## Example

See `examples/vector_env_example.py` for a complete working example.

## Limitations

1. **Single Transport**: All environments share the same transport, which may limit true parallelism
2. **Synchronized Operations**: All environments operate synchronously
3. **Shared State**: Some state may be shared between environments depending on the underlying implementation

## Future Enhancements

- **Async Support**: True asynchronous operation for better performance
- **Multiple Transports**: Support for multiple transport instances
- **Custom Batching**: More sophisticated batching strategies
- **Memory Optimization**: Better memory management for large numbers of environments 