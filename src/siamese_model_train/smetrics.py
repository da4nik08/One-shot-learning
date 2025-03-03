from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, average_precision_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


class Metrics():
    def __init__(self):
        self.all_actual = list()
        self.all_embeddings = []

    def batch_step(self, actualv, embeddings):
        self.all_actual.append(actualv.numpy(force=True))
        self.all_embeddings.append(embeddings.numpy(force=True))
    
    def get_metrics(self, train_emb, train_lbl, top_k=1):
        self.all_actual = np.concatenate(self.all_actual) 
        self.all_embeddings = np.concatenate(self.all_embeddings)           # collect all val data
        train_emb = train_emb.numpy(force=True)
        train_lbl = train_lbl.numpy(force=True)                             # train all data
        distance_matrix = pairwise_distances(self.all_embeddings, 
                                             np.concatenate([self.all_embeddings, train_emb]), 
                                             metric="euclidean", force_all_finite=False)  # distance matrix val embeddings to 
                                                                                          # database emb. Shape N_val , N_all
        np.fill_diagonal(distance_matrix, np.inf)                 # to prevent self-selection distance between same vector=0
        ranked_indices = np.argsort(distance_matrix, axis=1)      # Sort by distance
        all_labels = np.concatenate([self.all_actual, train_lbl]) # full database
        ranked_labels = all_labels[ranked_indices]
        
        #similarity_matrix = cosine_similarity(self.all_embeddings) # For cosine similarity metric None
        y_pred = list()
        for lbl in ranked_labels:
            if top_k == 3:
                if lbl[1] == lbl[2]:
                    y_pred.append(lbl[1])
                else:
                    y_pred.append(lbl[0])
            else:
                y_pred.append(lbl[0])      
                                                    # Knn for k<=3 top sample
        ap_scores = []
        for i in range(len(self.all_actual)):
            y_true = (ranked_labels[i][:-1] == self.all_actual[i]).astype(int)  # Binary relevance (ground_truth)
            y_score = -distance_matrix[i][ranked_indices[i]][:-1]
            ap_scores.append(average_precision_score(y_true, y_score))

        y_pred = np.array(y_pred)
        mAP = np.mean(ap_scores)
        recall = recall_score(self.all_actual, y_pred, average='weighted', zero_division=0)
        precision = precision_score(self.all_actual, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.all_actual, y_pred, average='weighted', zero_division=0)
        accuracy = accuracy_score(self.all_actual, y_pred)
        
        return recall, precision, f1, accuracy, mAP