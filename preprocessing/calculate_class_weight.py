from collections import defaultdict
from collections import Counter
import torch


def calculate_cw(train_labels):
    class_counts = Counter(train_labels)
    # Total number of instances
    total_instances = sum(class_counts.values())
    
    # Calculate class weights (inverse frequency)
    class_weights = {cls: total_instances / count for cls, count in class_counts.items()}
    
    # Normalize weights (optional)
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    
    # Convert class weights to a PyTorch tensor
    num_classes = len(class_counts)
    weight_tensor = torch.zeros(num_classes)
    for cls, weight in class_weights.items():
        weight_tensor[cls] = weight

    return class_weights