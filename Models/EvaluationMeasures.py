from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class EvaluationMeasures :
    # ---------- Initialization ---------- #
    def __init__(self):
        self.reset_counters()

    def reset_counters(self):
        self.predicted_labels = []
        self.correct_labels = []

        self.total_loss = 0
        self.number_of_steps = 0

    def set_correct_labels(self, labels_ls):
        self.correct_labels = labels_ls

    # ---------- Update ---------- #
    def append_label(self, label):
        self.predicted_labels.append(label)

    def append_correct_label(self, label):
        self.correct_labels.append(label)

    def append_loss(self, loss):
        self.total_loss = self.total_loss + loss
        self.number_of_steps = self.number_of_steps + 1

    # ---------- Evaluation measures ---------- #
    def compute_accuracy(self):
        return accuracy_score(y_true=self.correct_labels, y_pred=self.predicted_labels)

    def compute_precision(self):
        return precision_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_recall(self):
        return recall_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_f1score(self):
        return f1_score(y_true=self.correct_labels, y_pred=self.predicted_labels, average=None)

    def compute_loss(self):
        return self.total_loss / self.number_of_steps

    def compute_confusion_matrix(self):
        return confusion_matrix(self.correct_labels, self.predicted_labels)

