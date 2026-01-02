"""
Evaluation metrics for language models.
"""
from typing import List, Tuple, Dict
import math


def calculate_accuracy(predictions: List[int], targets: List[int]) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: List of predicted token indices
        targets: List of target token indices
        
    Returns:
        Accuracy as a percentage (0-100)
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return 0.0
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    return (correct / len(targets)) * 100.0


def calculate_top_k_accuracy(logits_list: List[List[float]], targets: List[int], k: int = 5) -> float:
    """
    Calculate top-k accuracy (if correct token is in top-k predictions).
    
    Args:
        logits_list: List of logit vectors for each prediction
        targets: List of target token indices
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as a percentage (0-100)
    """
    if not logits_list or not targets or len(logits_list) != len(targets):
        return 0.0
    
    correct = 0
    for logits, target in zip(logits_list, targets):
        # Get indices of top-k predictions
        top_k_indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:k]
        if target in top_k_indices:
            correct += 1
    
    return (correct / len(targets)) * 100.0


def calculate_precision_recall_f1(predictions: List[int], targets: List[int], vocab_size: int) -> Dict[str, float]:
    """
    Calculate macro-averaged precision, recall, and F1 score across all classes.
    
    Args:
        predictions: List of predicted token indices
        targets: List of target token indices
        vocab_size: Size of vocabulary
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate per-class metrics
    class_metrics = []
    
    for class_idx in range(vocab_size):
        # True positives: predicted and actual are both this class
        tp = sum(1 for pred, target in zip(predictions, targets) 
                if pred == class_idx and target == class_idx)
        
        # False positives: predicted this class but actual is different
        fp = sum(1 for pred, target in zip(predictions, targets) 
                if pred == class_idx and target != class_idx)
        
        # False negatives: actual is this class but predicted different
        fn = sum(1 for pred, target in zip(predictions, targets) 
                if pred != class_idx and target == class_idx)
        
        # Calculate precision and recall for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Only include classes that appear in targets (to avoid division by zero for rare classes)
        if tp + fn > 0:
            class_metrics.append({
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
    
    # Macro average across all classes
    if not class_metrics:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    avg_precision = sum(m["precision"] for m in class_metrics) / len(class_metrics)
    avg_recall = sum(m["recall"] for m in class_metrics) / len(class_metrics)
    avg_f1 = sum(m["f1"] for m in class_metrics) / len(class_metrics)
    
    return {
        "precision": avg_precision * 100.0,  # Convert to percentage
        "recall": avg_recall * 100.0,
        "f1": avg_f1 * 100.0
    }
