import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np

final_test_pt = torch.load("fine_tune/fine_tuned_result.pt")
final_test = final_test_pt.numpy()
labels_test = np.load("cluster_label_test.npy")


precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels_test, y_pred=final_test)

print("Done!")