from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score


class Metrics():
    def __init__(self):
        self.all_actual = np.array([])
        self.all_predicted = np.array([])

    def batch_step(self, actualv, predictedv):
        self.all_actual = np.concatenate([self.all_actual, actualv.numpy(force=True)])
        self.all_predicted = np.concatenate([self.all_predicted, predictedv.numpy(force=True)])

    def get_metrics(self):
        recall = recall_score(self.all_actual, self.all_predicted, average='weighted', zero_division=0)
        precision = precision_score(self.all_actual, self.all_predicted, average='weighted', zero_division=0)
        f1 = f1_score(self.all_actual, self.all_predicted, average='weighted', zero_division=0)
        accuracy = accuracy_score(self.all_actual, self.all_predicted)
        return recall, precision, f1, accuracy