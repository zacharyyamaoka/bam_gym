from ros_py_types.bam_msgs import ErrorCode, ErrorType

def ensure_list(var):
    if var is None:
        return []
    elif isinstance(var, (list, tuple)):
        return list(var)
    else:
        return [var]
        
def is_env_success(info):
    """
    Check if the environment step was successful based on the error code in the info dictionary.
    """
    if "header" in info and "error_code" in info["header"]:
        error_val = info["header"]["error_code"]["value"]
        return error_val == ErrorType.SUCCESS
    return False