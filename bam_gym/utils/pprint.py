import copy
import numpy as np
from bam_gym.ros_types.bam_msgs import ErrorType

def describe_ndarray(value):
    if isinstance(value, np.ndarray):
        return f"np.ndarray{value.shape}"
    return value

def recursive_format(obj):
    if isinstance(obj, dict):
        return {k: recursive_format(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_format(item) for item in obj]
    else:
        return describe_ndarray(obj)

def print_step_result(observation, action, reward, terminated, truncated, info):
    display_obs = recursive_format(copy.deepcopy(observation))
    display_info = recursive_format(copy.deepcopy(info))

    # Special formatting for error_code if it exists
    try:
        code_value = info["header"]["error_code"]["value"]
        display_info["header"]["error_code"]["value"] = ErrorType(code_value).name
    except Exception:
        pass  # Gracefully skip if structure doesn't match

    print(f"\nStep Result:")
    print(f"Action: {action}")
    print(f"Observation: {display_obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {display_info}")
