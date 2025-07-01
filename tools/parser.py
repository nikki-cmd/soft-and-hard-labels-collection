def parse_arg_value(value):
    try:
        return int(value)
    
    except ValueError:
        pass
    
    try:
        return float(value)
    
    except ValueError:
        pass
    
    if value.lower() in ['true', 'yes']:
        return True
    if value.lower() in ['false', 'no']:
        return False
        
    return value


def parse_key_value_args(args):
    result = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            result[key] = parse_arg_value(value)
    return result