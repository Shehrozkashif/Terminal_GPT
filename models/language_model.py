import random
from typing import List, Tuple, Dict, Union
import math
from nn.maths_operations import matrix_vector_multiply, add_vectors
from models.lstm import lstm_forward, create_initial_lstm_state, lstm_backward
from nn.activations import softmax
from nn.losses import cross_entropy_loss, cross_entropy_gradient


def initialize_language_model(vocab_size: int, embedding_dim: int, hidden_size: int, model_type: str = "lstm") -> dict:
    """
    Initialize a simple LSTM language model.
    
    Returns a dictionary with:
    - embeddings
    - params (LSTM weights and biases)
    - type
    """
    model = {"type": model_type, "vocab_size": vocab_size, "embedding_dim": embedding_dim, "hidden_size": hidden_size}

                          
                                        
    embeddings = [[random.uniform(-0.1, 0.1) for _ in range(embedding_dim)] for _ in range(vocab_size)]
    model["embeddings"] = embeddings

                               
                                
    input_size = embedding_dim

                                                                    
    def rand_matrix(rows, cols):
        return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]

                                        
    def rand_vector(size):
        return [random.uniform(-0.1, 0.1) for _ in range(size)]

    params = {
        "Wf": rand_matrix(hidden_size, hidden_size + input_size),
        "Wi": rand_matrix(hidden_size, hidden_size + input_size),
        "Wo": rand_matrix(hidden_size, hidden_size + input_size),
        "Wc": rand_matrix(hidden_size, hidden_size + input_size),
        "bf": rand_vector(hidden_size),
        "bi": rand_vector(hidden_size),
        "bo": rand_vector(hidden_size),
        "bc": rand_vector(hidden_size),
                                              
        "Wy": rand_matrix(vocab_size, hidden_size),
        "by": rand_vector(vocab_size),
    }

    model["params"] = params

    return model


def embed_sequence(indices: List[int], embeddings: List[List[float]]) -> List[List[float]]:
    """Lookup embeddings for token indices."""
    return [embeddings[idx] for idx in indices]


def language_model_forward(input_indices: List[int], model: Dict) -> Tuple[List[List[float]], Dict]:
    """Forward pass through language model."""
    embedded = embed_sequence(input_indices, model['embeddings'])
    
    if model['type'] == 'lstm':
        h_init, c_init = create_initial_lstm_state(model['hidden_size'])
        outputs, hiddens, cells, caches = lstm_forward(embedded, h_init, c_init, model['params'])
        cache = {'hiddens': hiddens, 'cells': cells, 'caches': caches, 'embedded': embedded}
    
    return outputs, cache



def language_model_loss(input_indices, target_indices, model):
    """
    Forward and backward pass for a language model.
    Returns average loss and gradients (including embeddings).
    """

                  
    outputs, cache = language_model_forward(input_indices, model)

    total_loss = 0.0
    output_grads = []

    for logits, target in zip(outputs, target_indices):
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, target)
        total_loss += loss
        grad = cross_entropy_gradient(probs, target)
        output_grads.append(grad)

    avg_loss = total_loss / len(outputs)

                   
    embedded = cache['embedded']
    if model['type'] == 'lstm':
        lstm_grads = lstm_backward(
            embedded, 
            cache['hiddens'], 
            cache['cells'], 
            cache['caches'], 
            output_grads, 
            model['params']
        )

                        
    param_grads = {k: lstm_grads[k] for k in model['params'].keys()}
    embedding_grads = lstm_grads['d_embeds']

    gradients = {
        'params': param_grads,
        'embeddings': embedding_grads
    }

    return avg_loss, gradients


def generate_text(
    model: Dict,
    start_tokens: List[int],
    max_length: int,
    temperature: float = 1.0,
    top_k: int = 0
) -> List[int]:
    """Generate text from language model."""

    generated = start_tokens.copy()

    for _ in range(max_length):
                                                                     
        outputs, _ = language_model_forward(
            generated[-min(50, len(generated)):],
            model
        )

        next_logits = outputs[-1]

                                   
        if temperature != 1.0:
            next_logits = [x / temperature for x in next_logits]

        probs = softmax(next_logits)

                        
        if top_k > 0:
            sorted_indices = sorted(
                range(len(probs)),
                key=lambda i: probs[i],
                reverse=True
            )
            top_k_indices = sorted_indices[:top_k]
            top_k_probs = [probs[i] for i in top_k_indices]

            total = sum(top_k_probs)
            top_k_probs = [p / total for p in top_k_probs]

            next_token = random.choices(
                top_k_indices,
                weights=top_k_probs,
                k=1
            )[0]
        else:
            next_token = random.choices(
                range(len(probs)),
                weights=probs,
                k=1
            )[0]

        generated.append(next_token)

                               
        if next_token == 3:
            break

    return generated
