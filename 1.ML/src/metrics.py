"""
Common evaluation metrics for Credit Card Fraud Detection
Primary Metric:   ROC-AUC
Secondary Metric: PR-AUC
"""
import numpy as np

def compute_roc_auc(y_true, y_pred_proba):
    """
    Computes Area Under the ROC Curve (ROC-AUC).
    Uses sklearn if available, otherwise fast exact Mann-Whitney U statistic via numpy.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred_proba))
    except ImportError:
        # Exact Mann-Whitney U calculation in pure numpy
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred_proba).ravel()
        
        pos_mask = (y_true == 1)
        n_pos = np.sum(pos_mask)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
            
        # Ranks of predictions (1-based)
        order = np.argsort(y_pred)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_pred) + 1)
        
        sum_pos_ranks = np.sum(ranks[pos_mask])
        auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)

def compute_pr_auc(y_true, y_pred_proba):
    """
    Computes Area Under the Precision-Recall Curve (PR-AUC / Average Precision).
    """
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred_proba))
    except ImportError:
        # Pure numpy calculation
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred_proba).ravel()
        
        order = np.argsort(-y_pred)
        y_true_sorted = y_true[order]
        
        tp = np.cumsum(y_true_sorted == 1)
        fp = np.cumsum(y_true_sorted == 0)
        
        precision = tp / (tp + fp)
        recall = tp / np.sum(y_true == 1)
        
        # Trapezoidal rule for PR-AUC
        # Add (0, 1) starting point
        recall_diff = np.diff(np.concatenate(([0.0], recall)))
        pr_auc = np.sum(precision * recall_diff)
        return float(pr_auc)

def evaluate_predictions(y_true, y_pred_proba, prefix=""):
    """
    Evaluates predictions and prints both ROC-AUC and PR-AUC.
    """
    roc_auc = compute_roc_auc(y_true, y_pred_proba)
    pr_auc = compute_pr_auc(y_true, y_pred_proba)
    
    header = f"{prefix} Evaluation Results" if prefix else "Evaluation Results"
    print(f"\n{'='*35}")
    print(f"   {header}")
    print(f"{'='*35}")
    print(f"ROC-AUC (Primary):   {roc_auc:.5f}")
    print(f"PR-AUC  (Secondary): {pr_auc:.5f}")
    print(f"{'='*35}\n")
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}

if __name__ == "__main__":
    # Quick sanity test
    y_test_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_test_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.05, 0.4])
    evaluate_predictions(y_test_true, y_test_pred, prefix="Sanity Check")
