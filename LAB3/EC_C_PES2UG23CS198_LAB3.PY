import torch

def get_entropy_of_dataset(data_tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset (for PyTorch)
    Last column in 'data_tensor' is assumed to be the target variable.
    """
    if data_tensor.shape[0] == 0:
        return 0.0
    labels = data_tensor[:, -1]
    classes, class_counts = torch.unique(labels, return_counts=True)
    total_samples = data_tensor.shape[0]
    probs = class_counts.float() / total_samples
    entropy_val = 0.0
    for p in probs:
        if p > 0:
            entropy_val -= p * torch.log2(p)
    return float(entropy_val)

def get_avg_info_of_attribute(data_tensor: torch.Tensor, attr_index: int):
    """
    Calculate the average information (weighted entropy) of an attribute (PyTorch).
    """
    if data_tensor.shape[0] == 0 or attr_index < 0 or attr_index >= data_tensor.shape[1] - 1:
        return 0.0
    col_values = data_tensor[:, attr_index]
    total_samples = data_tensor.shape[0]
    unique_vals = torch.unique(col_values)
    weighted_entropy = 0.0
    for val in unique_vals:
        subset_mask = (col_values == val)
        subset_tensor = data_tensor[subset_mask]
        subset_weight = subset_tensor.shape[0] / total_samples
        if subset_tensor.shape[0] > 0:
            subset_ent = get_entropy_of_dataset(subset_tensor)
            weighted_entropy += subset_weight * subset_ent
    return float(weighted_entropy)

def get_information_gain(data_tensor: torch.Tensor, attr_index: int):
    """
    Calculate information gain for an attribute in the dataset (PyTorch).
    """
    if data_tensor.shape[0] == 0:
        return 0.0
    dataset_entropy = get_entropy_of_dataset(data_tensor)
    avg_entropy = get_avg_info_of_attribute(data_tensor, attr_index)
    info_gain_val = dataset_entropy - avg_entropy
    return round(float(info_gain_val), 4)

def get_selected_attribute(data_tensor: torch.Tensor):
    """
    Return:
      - dict: {attribute_index: information_gain}
      - int: index of attribute with highest info gain
    for PyTorch tensors.
    """
    if data_tensor.shape[0] == 0 or data_tensor.shape[1] <= 1:
        return {}, -1
    num_attrs = data_tensor.shape[1] - 1
    info_gain_dict = {}
    for idx in range(num_attrs):
        info_gain_dict[idx] = get_information_gain(data_tensor, idx)
    if not info_gain_dict:
        return {}, -1
    best_attr = max(info_gain_dict, key=info_gain_dict.get)
    return info_gain_dict, best_attr

