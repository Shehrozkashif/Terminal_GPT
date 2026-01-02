import random
from typing import List, Tuple, Dict, Union
import math
from nn.maths_operations import Matrix, xavier_init, matrix_vector_multiply, zeros, add_vectors, Vector



def initialize_lstm_parameters(input_size: int, hidden_size: int,
                               output_size: int) -> Dict[str, Matrix]:
    """Initialize LSTM parameters."""
    params = {
        'Wf': xavier_init((input_size + hidden_size, hidden_size)),
        'bf': zeros((hidden_size,)),
        'Wi': xavier_init((input_size + hidden_size, hidden_size)),
        'bi': zeros((hidden_size,)),
        'Wc': xavier_init((input_size + hidden_size, hidden_size)),
        'bc': zeros((hidden_size,)),
        'Wo': xavier_init((input_size + hidden_size, hidden_size)),
        'bo': zeros((hidden_size,)),
        'Why': xavier_init((hidden_size, output_size)),
        'by': zeros((output_size,))
    }
    return params


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def lstm_cell_forward(x, h_prev, c_prev, params):
    """
    Single LSTM cell forward pass.
    x: input embedding (embedding_dim,)
    h_prev: previous hidden state (hidden_size,)
    c_prev: previous cell state (hidden_size,)
    params: dict of LSTM parameters
    Returns:
        h_next: next hidden state
        c_next: next cell state
        cache: dict for backward pass
    """

    hidden_size = len(h_prev)

                              
    concat = h_prev + x                                      

                          
    f_raw = matrix_vector_multiply(params['Wf'], concat)
    i_raw = matrix_vector_multiply(params['Wi'], concat)
    o_raw = matrix_vector_multiply(params['Wo'], concat)
    g_raw = matrix_vector_multiply(params['Wc'], concat)

                
    for j in range(hidden_size):
        f_raw[j] += params['bf'][j]
        i_raw[j] += params['bi'][j]
        o_raw[j] += params['bo'][j]
        g_raw[j] += params['bc'][j]

                 
    f = [sigmoid(v) for v in f_raw]
    i = [sigmoid(v) for v in i_raw]
    o = [sigmoid(v) for v in o_raw]
    g = [math.tanh(v) for v in g_raw]

                           
    c_next = [f[j] * c_prev[j] + i[j] * g[j] for j in range(hidden_size)]
    h_next = [o[j] * math.tanh(c_next[j]) for j in range(hidden_size)]

                        
    cache = {
        'h_prev': h_prev,
        'c_prev': c_prev,
        'x': x,
        'f': f,
        'i': i,
        'o': o,
        'g': g,
        'c_next': c_next,
        'concat': concat
    }

    return h_next, c_next, cache


def lstm_forward(inputs: List[Vector], h_init: Vector, c_init: Vector,
                params: Dict[str, Matrix]) -> Tuple[List[Vector], List[Vector], List[Vector], List[Dict]]:
    """Forward pass through LSTM sequence."""
    hidden_states = [h_init]
    cell_states = [c_init]
    outputs = []
    caches = []
    
    h = h_init
    c = c_init
    
    for x in inputs:
        h, c, cache = lstm_cell_forward(x, h, c, params)
        hidden_states.append(h)
        cell_states.append(c)
        caches.append(cache)
        
        y = matrix_vector_multiply(params['Wy'], h)
        y = add_vectors(y, params['by'])
        outputs.append(y)
    
    return outputs, hidden_states, cell_states, caches


def lstm_cell_backward(dh_next, dc_next, cache, params):
    """
    Single LSTM cell backward pass.
    dh_next: gradient of loss w.r.t next hidden state
    dc_next: gradient of loss w.r.t next cell state
    cache: dictionary from forward pass
    params: LSTM parameters
    Returns:
        d_x: gradient w.r.t input x
        dh_prev: gradient w.r.t previous hidden state
        dc_prev: gradient w.r.t previous cell state
        dWf, dWi, dWo, dWc: gradients of weight matrices
        dbf, dbi, dbo, dbc: gradients of biases
    """

    h_prev = cache['h_prev']
    c_prev = cache['c_prev']
    x = cache['x']
    f = cache['f']
    i = cache['i']
    o = cache['o']
    g = cache['g']
    c_next = cache['c_next']
    concat = cache['concat']

    hidden_size = len(h_prev)
    input_size = len(x)

                                
    tanh_c = [math.tanh(c_next[j]) for j in range(hidden_size)]

                              
    do = [dh_next[j] * tanh_c[j] for j in range(hidden_size)]

                             
    dc = [dc_next[j] + dh_next[j] * o[j] * (1 - tanh_c[j]**2) for j in range(hidden_size)]

                        
    df = [dc[j] * c_prev[j] * f[j] * (1 - f[j]) for j in range(hidden_size)]
    di = [dc[j] * g[j] * i[j] * (1 - i[j]) for j in range(hidden_size)]
    dg = [dc[j] * i[j] * (1 - g[j]**2) for j in range(hidden_size)]

                                  
    dWf = [[0.0 for _ in range(hidden_size + input_size)] for _ in range(hidden_size)]
    dWi = [[0.0 for _ in range(hidden_size + input_size)] for _ in range(hidden_size)]
    dWo = [[0.0 for _ in range(hidden_size + input_size)] for _ in range(hidden_size)]
    dWc = [[0.0 for _ in range(hidden_size + input_size)] for _ in range(hidden_size)]

    for i_row in range(hidden_size):
        for j_col in range(hidden_size + input_size):
            dWf[i_row][j_col] = df[i_row] * concat[j_col]
            dWi[i_row][j_col] = di[i_row] * concat[j_col]
            dWo[i_row][j_col] = do[i_row] * concat[j_col]
            dWc[i_row][j_col] = dg[i_row] * concat[j_col]

                         
    dbf = df.copy()
    dbi = di.copy()
    dbo = do.copy()
    dbc = dg.copy()

                                        
    dconcat = [0.0 for _ in range(hidden_size + input_size)]
    for i_row in range(hidden_size):
        for j_col in range(hidden_size + input_size):
            dconcat[j_col] += df[i_row] * params['Wf'][i_row][j_col]
            dconcat[j_col] += di[i_row] * params['Wi'][i_row][j_col]
            dconcat[j_col] += do[i_row] * params['Wo'][i_row][j_col]
            dconcat[j_col] += dg[i_row] * params['Wc'][i_row][j_col]

    dh_prev = dconcat[:hidden_size]
    d_x = dconcat[hidden_size:]
    dc_prev = [dc[j] * f[j] for j in range(hidden_size)]

    return d_x, dh_prev, dc_prev, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc


def lstm_backward(embedded, hiddens, cells, caches, output_grads, params):
    """
    Backpropagation through time for LSTM.
    Returns dictionary of gradients matching params.
    """

    hidden_size = len(hiddens[0])
    input_size = len(embedded[0])
    seq_len = len(embedded)

                                    
    grads = {}
    for key, val in params.items():
        if isinstance(val[0], list):             
            grads[key] = [[0.0 for _ in range(len(val[0]))] for _ in range(len(val))]
        else:             
            grads[key] = [0.0 for _ in range(len(val))]

    dh_next = [0.0 for _ in range(hidden_size)]
    dc_next = [0.0 for _ in range(hidden_size)]

    d_embeds = [[0.0 for _ in range(input_size)] for _ in range(seq_len)]

    for t in reversed(range(seq_len)):
                                    
        dy = output_grads[t]                     
        h = hiddens[t]

                                  
        for i in range(len(params['Wy'])):
            for j in range(hidden_size):
                grads['Wy'][i][j] += dy[i] * h[j]
        for i in range(len(params['by'])):
            grads['by'][i] += dy[i]

                              
        dh = [0.0 for _ in range(hidden_size)]
        for i in range(hidden_size):
            for j in range(len(params['Wy'])):
                dh[i] += params['Wy'][j][i] * dy[j]
            dh[i] += dh_next[i]

                                      
        dc = dc_next.copy()
        d_embed, dh_next, dc_next, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc = lstm_cell_backward(
            dh, dc, caches[t], params
        )

                                        
        for i in range(hidden_size):
            for j in range(hidden_size + input_size):
                grads['Wf'][i][j] += dWf[i][j]
                grads['Wi'][i][j] += dWi[i][j]
                grads['Wo'][i][j] += dWo[i][j]
                grads['Wc'][i][j] += dWc[i][j]
        for i in range(hidden_size):
            grads['bf'][i] += dbf[i]
            grads['bi'][i] += dbi[i]
            grads['bo'][i] += dbo[i]
            grads['bc'][i] += dbc[i]

                            
        d_embeds[t] = d_embed

                      
    grads['d_embeds'] = d_embeds
    return grads


def create_initial_lstm_state(hidden_size: int) -> Tuple[Vector, Vector]:
    """Create initial hidden and cell states."""
    return zeros((hidden_size,)), zeros((hidden_size,))

