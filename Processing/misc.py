import numpy as np
from scipy.linalg import null_space, norm

# Adapted from 01_classification... by Tiago
def calculate_performance_stats_by_class(confusion_mat: np.ndarray) -> list[dict[int, float], dict[int, float]]:
    sensitivity = np.zeros(confusion_mat.shape[0])
    specificity = np.zeros(confusion_mat.shape[0])
    accuracy = np.zeros(confusion_mat.shape[0])
    fpr = np.zeros(confusion_mat.shape[0])
    total_dataset_count = np.zeros(confusion_mat.shape[0])

    # Total number of classes
    num_classes = len(confusion_mat)

    for class_idx in range(num_classes):
        # True Positives: Diagonal element of the confusion matrix
        tp = confusion_mat[class_idx, class_idx]

        # False Negatives: Sum of the row for the class, minus tp
        fn = np.sum(confusion_mat[class_idx, :]) - tp

        # True Negatives: Sum of all elements, minus the sum of the row and the column for the class, plus tp
        tn = np.sum(confusion_mat) - np.sum(confusion_mat[class_idx, :]) - np.sum(
            confusion_mat[:, class_idx]) + tp

        # False Positives: Sum of the column for the class, minus tp
        fp = np.sum(confusion_mat[:, class_idx]) - tp

        # Calculate sensitivity (recall) and specificity
        sensitivity[class_idx] = round(tp / (tp + fn), 3) if (tp + fn) != 0 else 0
        specificity[class_idx] = round(tn / (tn + fp), 3) if (tn + fp) != 0 else 0
        accuracy[class_idx] = round((tp + tn) / (tp + tn + fp + fn), 3) if (tp + tn + fp + fn) != 0 else 0
        fpr[class_idx] = round(fp / (fp + tn), 3) if (fp + tn) != 0 else 0
        total_dataset_count[class_idx] = np.sum(confusion_mat[class_idx, :])
    mean_dataset_count = np.mean(total_dataset_count)
    prop_by_class = mean_dataset_count/ total_dataset_count
    overall_accuracy = np.diag(confusion_mat*np.diag(prop_by_class)).sum() / np.sum(total_dataset_count*prop_by_class)


    return sensitivity, specificity, accuracy, fpr, overall_accuracy

def return_positive_negative_counts_all_classes(accuracy, sensitivity, specificity, fpr):
    for i in range(len(accuracy)):
        a = accuracy[i]
        se = sensitivity[i]
        sp = specificity[i]
        f = fpr[i]

        mat = np.array([[a-1, a-1, a, 1],[se-1, 0,0, se],[0,sp-1,sp,0],[0,f,f-1,0]])
        ns = null_space(mat)
        norm_ns = ns/np.sum(ns)
        print(norm_ns)