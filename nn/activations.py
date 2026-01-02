import math
from typing import List, Union
from nn.maths_operations import Vector


def sigmoid(x: float) -> float:
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    if x > 20:
        return 1.0
    elif x < -20:
        return -1.0
    exp_2x = math.exp(2 * x)
    return (exp_2x - 1.0) / (exp_2x + 1.0)


def tanh_derivative(x: float) -> float:
    """Derivative of tanh."""
    t = tanh(x)
    return 1.0 - t * t


def softmax(vector: Vector) -> Vector:
    """Softmax activation (numerically stable)."""
    max_val = max(vector)
    exp_values = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_values)
    return [e / sum_exp for e in exp_values]

