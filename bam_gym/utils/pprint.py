import copy
import numpy as np
from bam_gym.ros_types.bam_msgs import ErrorType
from gymnasium import spaces
import json

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

def print_step_result(i, observation, action, reward, terminated, truncated, info):
    display_obs = recursive_format(copy.deepcopy(observation))
    display_info = recursive_format(copy.deepcopy(info))

    # Special formatting for error_code if it exists
    try:
        code_value = info["header"]["error_code"]["value"]
        display_info["header"]["error_code"]["value"] = ErrorType(code_value).name
    except Exception:
        pass  # Gracefully skip if structure doesn't match

    print(f"\n[{i}] Step Result:")
    print(f"Action: {action}")
    print(f"Observation: {display_obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {display_info}")


def to_jsonable(obj):
    """
    Recursively convert a Gym sample to a JSON-serializable Python structure.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj
    
def print_sampled_action(action):

    json_obj = to_jsonable(action)
    print(json.dumps(json_obj, indent=2))

def format_space(space, indent=0):
    prefix = "  " * indent
    if isinstance(space, spaces.Dict):
        items = []
        for key, val in space.spaces.items():
            items.append(f'{prefix}"{key}": {format_space(val, indent + 1)}')
        return "{\n" + ",\n".join(items) + f"\n{prefix}}}"
    elif isinstance(space, spaces.Sequence):
        return f"[{format_space(space.feature_space, indent + 1)}]"
    elif isinstance(space, spaces.Box):
        return (f'Box(low={space.low if space.shape else space.low.item()}, '
                f'high={space.high if space.shape else space.high.item()}, '
                f'shape={space.shape}, dtype="{space.dtype}")')
    elif isinstance(space, spaces.Text):
        return f'Text(max_length={space.max_length})'
    else:
        return str(space)
    
def print_gym_space(space):
    print(format_space(space))

