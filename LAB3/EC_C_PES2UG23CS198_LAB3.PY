import torch

def get_entropy_of_dataset(tensor: torch.Tensor) -> float:
    """
    Compute entropy of dataset (last column is target).
    """
    if tensor.shape[0] == 0:
        return 0.0
    
    labels = tensor[:, -1]
    _, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum()

    # Compact form using sum comprehension
    return float(sum([-p * torch.log2(p) for p in probs if p > 0]))


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int) -> float:
    """
    Compute weighted average entropy of an attribute.
    """
    if tensor.shape[0] == 0 or attribute < 0 or attribute >= tensor.shape[1] - 1:
        return 0.0
    
    values = tensor[:, attribute]
    total = tensor.shape[0]

    # Weighted entropy via list comprehension
    return float(sum(
        (subset.shape[0] / total) * get_entropy_of_dataset(subset)
        for val in torch.unique(values)
        for subset in [tensor[values == val]]
        if subset.shape[0] > 0
    ))


def get_information_gain(tensor: torch.Tensor, attribute: int) -> float:
    """
    Compute information gain of an attribute.
    """
    if tensor.shape[0] == 0:
        return 0.0
    
    return round(
        get_entropy_of_dataset(tensor) - get_avg_info_of_attribute(tensor, attribute), 4
    )


def get_selected_attribute(tensor: torch.Tensor):
    """
    Returns:
      - dict {attribute: information_gain}
      - best attribute index
    """
    if tensor.shape[0] == 0 or tensor.shape[1] <= 1:
        return {}, -1

    # Dictionary comprehension for info gains
    gains = {i: get_information_gain(tensor, i) for i in range(tensor.shape[1] - 1)}
    
    return (gains, max(gains, key=gains.get)) if gains else ({}, -1)
