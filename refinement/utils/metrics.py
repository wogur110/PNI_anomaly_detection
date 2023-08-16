"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_confusion_matrix(anomaly_ground_truth_labels, anomaly_prediction_weights, thresholds):

    anomaly_ground_truth_labels = np.asarray(anomaly_ground_truth_labels)
    TP, FP, FN, TN = [], [], [], []
    for index in range(len(thresholds)):
        current_threshold = thresholds[index]
        anomaly_prediction_labels = (anomaly_prediction_weights >= current_threshold)

        current_TP = np.mean((anomaly_ground_truth_labels == True) & (anomaly_prediction_labels == True))
        current_FP = np.mean((anomaly_ground_truth_labels == False) & (anomaly_prediction_labels == True))
        current_FN = np.mean((anomaly_ground_truth_labels == True) & (anomaly_prediction_labels == False))
        current_TN = np.mean((anomaly_ground_truth_labels == False) & (anomaly_prediction_labels == False))

        TP.append(current_TP)
        FP.append(current_FP)
        FN.append(current_FN)
        TN.append(current_TN)
    TP = np.asarray(TP)
    FP = np.asarray(FP)
    FN = np.asarray(FN)
    TN = np.asarray(TN)

    return TP, FP, FN, TN

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    try:
        fpr, tpr, thresholds = metrics.roc_curve(
            anomaly_ground_truth_labels, anomaly_prediction_weights
        )
    except:
        a=0
    thresholds = np.asarray(list(range(1000,500,-20)) + list(range(500,200,-10)) + list(range(200,100,-5)) + list(range(100,-1,-1)), dtype=float)/1000
    TP, FP, FN, TN = get_confusion_matrix(anomaly_ground_truth_labels, anomaly_prediction_weights, thresholds)
    precision = (TP) / (TP + FP + 1e-10)
    recall = (TP) / (TP + FN + 1e-10)
    F1 = 2 / (1/(precision+1e-10) + 1/(recall+1e-10))

    try:
        auroc = metrics.roc_auc_score(
            anomaly_ground_truth_labels, anomaly_prediction_weights
        )
    except:
        auroc = 0
    #return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}
    return {
        "auroc": auroc,
        "thresholds": thresholds,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "F1": F1,
    }

def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    thresholds = np.asarray(list(range(1000,500,-20)) + list(range(500,200,-10)) + list(range(200,100,-5)) + list(range(100,-1,-1)), dtype=float)/1000
    TP, FP, FN, TN = get_confusion_matrix(ground_truth_masks.squeeze(), anomaly_segmentations, thresholds)
    precision = (TP) / (TP + FP + 1e-10)
    recall = (TP) / (TP + FN + 1e-10)
    F1 = 2 / (1/(precision+1e-10) + 1/(recall+1e-10))

    precision_old, recall_old, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision_old * recall_old,
        precision_old + recall_old,
        out=np.zeros_like(precision_old),
        where=(precision_old + recall_old) != 0,
    )


    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "thresholds": thresholds,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "F1": F1,
    }
    """
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
    """