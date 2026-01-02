from typing import List, Dict, Tuple, Union, Optional
from collections import Counter


def build_vocab_from_tokens(tokens: List[str], min_freq: int = 1, 
                            max_vocab_size: Optional[int] = None) -> Dict[str, int]:
    """Build vocabulary from token list."""
    token_counts = Counter(tokens)
    filtered_tokens = [tok for tok, count in token_counts.items() 
                      if count >= min_freq]
    sorted_tokens = sorted(filtered_tokens, 
                          key=lambda x: token_counts[x], 
                          reverse=True)
    if max_vocab_size:
        sorted_tokens = sorted_tokens[:max_vocab_size - 4]
    
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<bos>': 2,
        '<eos>': 3
    }
    
    for idx, token in enumerate(sorted_tokens, start=4):
        vocab[token] = idx
    
    return vocab


def create_reverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    """Create index-to-token mapping from vocabulary."""
    return {idx: token for token, idx in vocab.items()}


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """Convert tokens to indices using vocabulary."""
    unk_idx = vocab.get('<unk>', 1)
    return [vocab.get(token, unk_idx) for token in tokens]


def decode_indices(indices: List[int], reverse_vocab: Dict[int, str]) -> List[str]:
    """Convert indices back to tokens."""
    unk_token = reverse_vocab.get(1, '<unk>')
    return [reverse_vocab.get(idx, unk_token) for idx in indices]


def get_vocab_size(vocab: Dict[str, int]) -> int:
    """Get vocabulary size."""
    return len(vocab)


def save_vocab(vocab: Dict[str, int], filepath: str) -> None:
    """Save vocabulary to file."""
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    with open(filepath, 'w', encoding='utf-8') as f:
        for token, idx in sorted_items:
            f.write(f"{token}\t{idx}\n")


def load_vocab(filepath: str) -> Dict[str, int]:
    """Load vocabulary from file."""
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, idx = parts
                    vocab[token] = int(idx)
    return vocab
