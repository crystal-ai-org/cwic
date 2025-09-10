

def str_to_int_list(s, sep=','):
    return [int(x.strip()) for x in s.split(sep) if x.strip().isdigit()]
