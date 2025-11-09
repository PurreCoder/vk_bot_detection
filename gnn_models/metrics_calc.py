import numpy as np


def compute_metrics(y_true, y_pred):
    correct = (y_pred == y_true).sum()
    accuracy = int(correct) / len(y_true)

    # TP, TN, FP, FN
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return {
        'accuracy': accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix
    }



