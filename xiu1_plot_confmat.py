from numpy.lib.function_base import average
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

final_test = np.load("cluster_final_test.npy")
labels_test = np.load("cluster_label_test.npy")

cof_mat = confusion_matrix(
    y_true=labels_test,
    y_pred=final_test
)

precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels_test, y_pred=final_test, average="macro")

print("Done!")