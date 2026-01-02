import random
import math
from typing import List, Tuple, Union


Matrix = List[List[float]]
Vector = List[float]


def zeros(shape: Tuple[int, ...]) -> Union[Vector, Matrix]:
    """Create matrix/vector of zeros."""
    if len(shape) == 1:
        return [0.0] * shape[0]
    elif len(shape) == 2:
        return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def random_uniform(shape: Tuple[int, ...], low: float = -0.1, 
                   high: float = 0.1) -> Union[Vector, Matrix]:
    """Create matrix with random uniform values."""
    if len(shape) == 1:
        return [random.uniform(low, high) for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[random.uniform(low, high) for _ in range(shape[1])] 
                for _ in range(shape[0])]


def random_normal(shape: Tuple[int, ...], mean: float = 0.0, 
                 std: float = 0.01) -> Union[Vector, Matrix]:
    """Create matrix with random normal values using Box-Muller."""
    def sample_normal():
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + z0 * std
    
    if len(shape) == 1:
        return [sample_normal() for _ in range(shape[0])]
    elif len(shape) == 2:
        return [[sample_normal() for _ in range(shape[1])] 
                for _ in range(shape[0])]


def xavier_init(shape: Tuple[int, int]) -> Matrix:
    """Xavier/Glorot initialization for weights."""
    fan_in, fan_out = shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random_uniform(shape, -limit, limit)


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Matrix multiplication: C = A @ B"""
    m, n = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    if n != n2:
        raise ValueError(f"Incompatible shapes: ({m},{n}) @ ({n2},{p})")
    result = zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result


def matrix_vector_multiply(matrix: Matrix, vector: Vector) -> Vector:
    """Matrix-vector multiplication: y = A @ x"""
    m, n = len(matrix), len(matrix[0])
    if len(vector) != n:
        raise ValueError(f"Incompatible shapes")
    result = zeros((m,))
    for i in range(m):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]
    return result


def transpose(matrix: Matrix) -> Matrix:
    """Transpose matrix."""
    rows, cols = len(matrix), len(matrix[0])
    result = zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result


def add_matrices(a: Matrix, b: Matrix) -> Matrix:
    """Element-wise matrix addition."""
    rows, cols = len(a), len(a[0])
    result = zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i][j] = a[i][j] + b[i][j]
    return result


def add_vectors(a: Vector, b: Vector) -> Vector:
    """Element-wise vector addition."""
    return [a[i] + b[i] for i in range(len(a))]


def scalar_multiply(matrix: Union[Matrix, Vector], scalar: float) -> Union[Matrix, Vector]:
    """Multiply matrix/vector by scalar."""
    if isinstance(matrix[0], list):
        return [[matrix[i][j] * scalar for j in range(len(matrix[0]))] 
                for i in range(len(matrix))]
    else:
        return [x * scalar for x in matrix]


def element_wise_multiply(a: Union[Matrix, Vector], 
                         b: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
    """Element-wise multiplication (Hadamard product)."""
    if isinstance(a[0], list):
        rows, cols = len(a), len(a[0])
        return [[a[i][j] * b[i][j] for j in range(cols)] for i in range(rows)]
    else:
        return [a[i] * b[i] for i in range(len(a))]


def outer_product(a: Vector, b: Vector) -> Matrix:
    """Compute outer product: a ⊗ b"""
    m, n = len(a), len(b)
    result = zeros((m, n))
    for i in range(m):
        for j in range(n):
            result[i][j] = a[i] * b[j]
    return result


def argmax(vector: Vector) -> int:
    """Return index of maximum value."""
    return max(range(len(vector)), key=lambda i: vector[i])

