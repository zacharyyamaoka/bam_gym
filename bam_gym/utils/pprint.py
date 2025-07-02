import copy
import numpy as np
from ros_py_types.bam_msgs import ErrorType
from gymnasium import spaces
import json

def describe_value(value):
    if isinstance(value, np.ndarray):
        if value.shape:
            return f"np.ndarray{value.shape}"
        else:
            return f"{value.item():.1f}"  # scalar ndarray
    return value

def recursive_format(obj):
    if isinstance(obj, dict):
        return {k: recursive_format(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [recursive_format(item) for item in obj]
    else:
        return describe_value(obj)

def format_observation(observation):
    formatted_obs = recursive_format(copy.deepcopy(observation))
    return formatted_obs

def format_info(info):
    formatted_info = recursive_format(copy.deepcopy(info))
    try:
        code_value = info["header"]["error_code"]["value"]
        formatted_info["header"]["error_code"]["value"] = ErrorType(code_value).name # type: ignore
    except Exception:
        pass  # 

    return formatted_info

def print_observation(observation):
    display_obs = format_observation(observation)
    print(f"\nObservation:")
    print(f"    {display_obs}")

def print_reset(observation, info):

    display_obs = format_observation(observation)
    display_info = format_info(info)

    print(f"\nReset Result:")
    print(f"    Observation : {display_obs}")
    print(f"    Info        : {display_info}")


def print_step(action, observation, reward, terminated, truncated, info, i=0):
    display_obs = format_observation(observation)
    display_info = format_info(info)

    print(f"\n[{i}] Step Result:")
    print(f"    Action      : {action}")
    print(f"    Observation : {display_obs}")
    print(f"    Reward      : {reward}")
    print(f"    Terminated  : {terminated}")
    print(f"    Truncated   : {truncated}")
    print(f"    Info        : {display_info}")


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
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects like Grasp objects
        try:
            return {k: to_jsonable(v) for k, v in obj.__dict__.items()}
        except Exception:
            # Fallback: convert to string representation
            return str(obj)
    else:
        return obj
    
def print_action(action):

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

