

def mpersec(mph):
    return mph * 0.44704

def mph(mpersec):
    return mpersec * 2.23694

def print_dict(d, indent=0):
   for key, value in d.items():
        print('\t' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
