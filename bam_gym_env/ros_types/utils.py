
def ensure_list(var):
    if var is None:
        return []
    elif isinstance(var, (list, tuple)):
        return list(var)
    else:
        return [var]
        