import random
from typing import Dict, List, Union
from nn.maths_operations import add_vectors, add_matrices, scalar_multiply


def sgd_step(params, gradients, learning_rate: float):
    """Stochastic Gradient Descent update."""
    if isinstance(params[0], list):
        return add_matrices(params, scalar_multiply(gradients, -learning_rate))
    else:
        return add_vectors(params, scalar_multiply(gradients, -learning_rate))


def clip_gradients(gradients: Dict, max_norm: float) -> Dict:
    """Clip gradients by global norm."""
    total_norm = 0.0
    for grad in gradients.values():
        if isinstance(grad[0], list):
            for row in grad:
                for val in row:
                    total_norm += val ** 2
        else:
            for val in grad:
                total_norm += val ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        clipped = {}
        for name, grad in gradients.items():
            clipped[name] = scalar_multiply(grad, scale)
        return clipped
    return gradients


def learning_rate_schedule(initial_lr: float, epoch: int, 
                          strategy: str = 'constant', **kwargs) -> float:
    """Learning rate scheduling."""
    if strategy == 'constant':
        return initial_lr
    elif strategy == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.95)
        return initial_lr * (decay_rate ** epoch)
    else:
        return initial_lr

