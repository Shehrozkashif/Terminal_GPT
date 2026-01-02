import random
import pickle
import sys
import time
from typing import List, Tuple, Union, Dict, Any

def load_text_file(filepath: str) -> str:
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def create_language_model_sequences(indices: List[int], seq_length: int) -> List[Tuple[List[int], List[int]]]:
    """Create (input, target) sequences for language modeling."""
    sequences = []
    for i in range(len(indices) - seq_length):
        input_seq = indices[i:i+seq_length]
        target_seq = indices[i+1:i+seq_length+1]
        sequences.append((input_seq, target_seq))
    return sequences

def create_batches(data: List, batch_size: int, shuffle: bool = True) -> List[List]:
    """Create batches from data."""
    data_copy = data.copy()
    if shuffle:
        random.shuffle(data_copy)
    batches = []
    for i in range(0, len(data_copy), batch_size):
        batch = data_copy[i:i+batch_size]
        batches.append(batch)
    return batches

def split_train_val(data: List, val_split: float = 0.1, 
                    shuffle: bool = True) -> Tuple[List, List]:
    """Split data into training and validation sets."""
    data_copy = data.copy()
    if shuffle:
        random.shuffle(data_copy)
    val_size = int(len(data_copy) * val_split)
    val_data = data_copy[:val_size]
    train_data = data_copy[val_size:]
    return train_data, val_data

def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def log_message(message: str) -> None:
    """Print message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def print_progress(current: int, total: int, prefix: str = "") -> None:
    """Print progress bar."""
    percent = float(current) * 100 / total
    bar_length = 20
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {percent:.2f}%")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
