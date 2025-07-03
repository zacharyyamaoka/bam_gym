import gymnasium as gym
from gymnasium import spaces
from abc import ABC
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List, Generic, Tuple
import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
ArrayType = TypeVar("ArrayType")

""" Design Notes:
    if terminate:
        obs, info = env.reset()
        
    action, policy_info = policy(obs, terminated, reward, info)
    obs, reward, terminated, truncated, info = env.step(action)

    class Env

        def reset()
            obs, obs_info = obs_client.step(action, action_info, reward=None)
            if scenario_client:
                scenario_client.reset()
            return obs, obs_info
                
        def step(action)
        
            action_info = action_client.step(action)
            obs, obs_info = obs_client.step(action, action_info, reward=None)
            reward, reward_info = reward_client.step(action, action_info, obs, obs_info)
            
            info = action_info + obs_info + reward_info
            terminate = False # do some simple logic here to determine, 
            return obs, reward, terminated, truncated, info
        
"""
# for simplicity let me have two types, vector env and non vector!

class VectorPolicy(Generic[ObsType, ActType, ArrayType]):
    pass

class Policy(Generic[ObsType, ActType]):
    """
    Generic policy that takes observations of type ObsType and returns actions of type ActType
    """
    def __init__(self):
        self.ready = False

    def init_env(self, env: gym.Env) -> None:
        """
        Initialize the policy with the environment.
        This can be used to set up any necessary state or parameters.
        """
        self.env = env
        self.action_space: spaces.Space[ActType] = env.action_space
        self.observation_space: spaces.Space[ObsType] = env.observation_space
        self.ready = True


    def __call__(self,
            observation: ObsType,  # Now properly typed!
            terminated: Optional[bool] = None,
            truncated: Optional[bool] = None,
            info: Optional[dict[str, Any]] = None,
            ) -> Tuple[ActType, dict[str, Any]]:  # Returns action and policy info
        
        assert self.ready
        # Implement your policy logic here

        action = self.action_space.sample()
        policy_info = {}
        
        return action, policy_info

class ActionClient(Generic[ActType]):
    def __init__(self):
        pass

    def init_env(self, env: gym.Env) -> None:
        self.env = env
        self.action_space: spaces.Space[ActType] = env.action_space

    def step(self, action: ActType) -> dict[str, Any]:
        return {}

class ObsClient(Generic[ObsType, ActType]):

    def __init__(self):
        pass

    def init_env(self, env: gym.Env) -> None:
        self.env = env
        self.action_space: spaces.Space[ActType] = env.action_space
        self.observation_space: spaces.Space[ObsType] = env.observation_space

    def reset(self) -> Tuple[ObsType, dict[str, Any]]:
        return self.observation_space.sample(), {}

    def query(self, action: ActType, action_info: dict[str, Any], reward: Optional[float] = None) -> Tuple[ObsType, dict[str, Any]]:
        return self.reset()  # Default implementation, can be overridden

class RewardClient(Generic[ObsType, ActType]):
    def __init__(self):
        pass

    def init_env(self, env: gym.Env) -> None:
        self.env = env
        self.action_space: spaces.Space[ActType] = env.action_space
        self.observation_space: spaces.Space[ObsType] = env.observation_space

    def query(self, action: Optional[ActType] = None,
                    action_info: Optional[dict[str, Any]] = None,
                    obs: Optional[ObsType] = None,
                    obs_info: Optional[dict[str, Any]] = None
                    ) -> Tuple[float, dict[str, Any]]:
        

        return 0.0, {}
    
    def _check_for_errors(self, action_info: dict[str, Any], obs_info: dict[str, Any]) -> None:
        """
        Check for any errors in the action or observation info.
        This can be overridden by subclasses to implement specific error checks.
        """
        pass

# Just simple triggers... actually perhaps these steps should return client responses?
class ScenarioClient():
    def __init__(self):
        pass

    def init_env(self, env: gym.Env) -> None:
        self.env = env   

    def reset(self) -> None:
        pass
       
    def step(self) -> None:
        pass

class VectorEnv(gym.vector.VectorEnv, Generic[ObsType, ActType, ArrayType]):

    def __init__(self, action_client, obs_client, reward_client, scenario_client=None):
        self.action_client = action_client
        self.obs_client = obs_client
        self.reward_client = reward_client
        self.scenario_client = scenario_client

    def step(self, actions: ActType) -> Tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        pass

class Env(gym.Env, Generic[ObsType, ActType]):

    def __init__(self, action_client: ActionClient,
                       obs_client: ObsClient,
                       reward_client: RewardClient,
                       scenario_client: ScenarioClient = None):
        
        self.action_client = action_client
        self.obs_client = obs_client
        self.reward_client = reward_client
        self.scenario_client = scenario_client

        self._init_observation_space()
        self._init_action_space()
        self._init_render()

        self._init_clients() # after you finish intializing env
    def _init_clients(self):
        self.action_client.init_env(self)
        self.obs_client.init_env(self)
        self.reward_client.init_env(self)
        if self.scenario_client:
            self.scenario_client.init_env(self)

    def _init_observation_space(self):
        self.observation_space = spaces.Dict({})
        self.observation_space["obs"] = spaces.Sequence(spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32))

    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(1) 

    def _init_render(self):
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, dict[str, Any]]:

        obs, obs_info = self.obs_client.reset()
        if self.scenario_client:
            self.scenario_client.reset()

        return obs, obs_info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict[str, Any]]:

        action_info = self.action_client.step(action)
        obs, obs_info = self.obs_client.query(action, action_info, reward=None)
        reward, reward_info = self.reward_client.query(action, action_info, obs, obs_info)

        info = {**action_info, **obs_info, **reward_info}
        terminated, truncated = self._check_done(obs, reward, info)                    
        return obs, reward, terminated, truncated, info

        action_info = self.action_client.step(actions)

    def _check_done(self, obs, reward, info) -> Tuple[bool, bool]:

        terminated = False
        truncated = False   
        return terminated, truncated
    
# Usage example showing how the generics work together
if __name__ == "__main__":

    policy = Policy()     

    action_client = ActionClient[int]()
    obs_client = ObsClient[np.ndarray, int]()
    reward_client = RewardClient[np.ndarray, int]()
    scenario_client = ScenarioClient()
    env = Env[np.ndarray, int](action_client, obs_client, reward_client, scenario_client)
    # env = gym.make("CartPole-v1")  # Example environment
    policy.init_env(env)

    obs, info = env.reset()
    terminated, truncated = None, None 

    for i in range(10):
        action, policy_info = policy(obs, terminated, truncated, info)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

# Make a helper that checks an info dict for a api Header to check for any issues, etc...