import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from utilities import load_config, save_pkl 


def siamese_classification(pred_embeddings, db_embeddings, db_labels):
    pred_embeddings = pred_embeddings.numpy(force=True)
    db_embeddings = db_embeddings.reshape(150, 1)
    distance_matrix = pairwise_distances(pred_embeddings, db_embeddings, metric="euclidean", force_all_finite=False)
    np.fill_diagonal(distance_matrix, np.inf)                 # to prevent self-selection distance between same vector=0
    ranked_indices = np.argsort(distance_matrix, axis=1)      # Sort by distance
    ranked_labels = db_labels[ranked_indices]
        
    y_pred = list()
    for lbl in ranked_labels:
        if top_k == 3:
            if lbl[1] == lbl[2]:
                y_pred.append(lbl[1])
            else:
                y_pred.append(lbl[0])
        else:
            y_pred.append(lbl[0])       # Knn for k<=3 top sample

    y_pred = np.array(y_pred)
    return y_pred