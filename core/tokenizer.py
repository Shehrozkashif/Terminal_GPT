import re
from typing import List

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple punctuation rules.
    """
    # Simple split by punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_word_level(text: str) -> List[str]:
    """
    Tokenize text into words, keeping punctuation as separate tokens.
    """
    # Add spaces around punctuation
    text = re.sub(r'([.,!?;:"\(\)\[\]])', r' \1 ', text)
    # Split by whitespace
    return text.split()

def detokenize(tokens: List[str]) -> str:
    """
    Convert tokens back to string.
    """
    text = " ".join(tokens)
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:"\(\)\[\]])', r'\1', text)
    return text
