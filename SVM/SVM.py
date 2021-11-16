from sklearn import svm
import Utils

def exe():
    model = svm.SVC()

    training_df = Utils.load_split(Utils.Split.TRAINING)

    labels_ls = Utils.get_labels(training_df)





