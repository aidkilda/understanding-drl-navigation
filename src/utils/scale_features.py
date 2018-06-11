def normalize(features, max_values, min_values):
    values_diff = max_values - min_values
    values_diff[values_diff == 0] = 1
    return (features - min_values) / values_diff

def standardize(features, mean, std):
    std[std == 0] = 1
    return (features - mean) / std



