import torch
import torch.nn as nn
import torch.nn.functional as F


def TripletLoss(loss_fn, embeddings, labels, mode='train', batch_size=None):
    distances = torch.cdist(embeddings, embeddings, p=2)  # Pairwise Euclidean distances
    labels = labels.unsqueeze(1)  # Shape: [N, 1]
    mask_positive = labels == labels.T  # Positive mask
    mask_negative = ~mask_positive  # Negative mask

    # Hard positives: Max distance within the positive mask
    distances_positive = distances.clone()
    distances_positive[~mask_positive] = -float('inf')  # Ignore non-positives
    hard_positive_indices = distances_positive.argmax(dim=1)
    # Hard negatives: Min distance within the negative mask
    distances_negative = distances.clone()
    distances_negative[~mask_negative] = float('inf')  # Ignore non-negatives
    hard_negative_indices = distances_negative.argmin(dim=1)

    anchor = embeddings
    positive = embeddings[hard_positive_indices]  # Hard positive embeddings
    negative = embeddings[hard_negative_indices]
    
    if mode == 'val':
        anchor = anchor[:batch_size]
        positive = positive[:batch_size]
        negative = negative[:batch_size]
    
    return loss_fn(anchor, positive, negative)