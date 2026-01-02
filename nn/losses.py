import math
from nn.maths_operations import Vector


def cross_entropy_loss(predictions: Vector, target_idx: int) -> float:
    """Cross-entropy loss for classification."""
    epsilon = 1e-10
    pred = max(epsilon, min(1.0 - epsilon, predictions[target_idx]))
    return -math.log(pred)


def cross_entropy_gradient(predictions: Vector, target_idx: int) -> Vector:
    """Gradient of cross-entropy loss w.r.t. logits."""
    gradient = predictions.copy()
    gradient[target_idx] -= 1.0
    return gradient


def perplexity_from_loss(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

