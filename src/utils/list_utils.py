def flatten(list):
    """ Flattens a list of lists into a list."""
    return [item for sublist in list for item in sublist]