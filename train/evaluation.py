import time
from typing import Tuple, Dict
from core.dataset import load_text_file, create_language_model_sequences, create_batches, load_pickle, log_message
from core.tokenizer import tokenize_word_level, split_sentences
from core.vocabulary import encode_tokens, load_vocab
from models.language_model import language_model_forward, language_model_loss
from nn.losses import perplexity_from_loss
from core.utils import format_time

def evaluate_model(
    model_path: str,
    data_path: str,
    vocab_path: str = "vocab.txt",
    batch_size: int = 16,
    seq_length: int = 30
) -> Dict[str, float]:
    """
    Evaluate a trained language model on a dataset.
    
    Args:
        model_path: Path to the saved model pickle file
        data_path: Path to the text file to evaluate on
        vocab_path: Path to the vocabulary file
        batch_size: Batch size for evaluation
        seq_length: Sequence length for evaluation
        
    Returns:
        Dictionary containing evaluation metrics (loss, perplexity)
    """
    log_message(f"Loading model from {model_path}...")
    model = load_pickle(model_path)
    
    log_message(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    
    log_message(f"Loading and preprocessing data from {data_path}...")
    text = load_text_file(data_path)
    sentences = split_sentences(text)
    
    all_tokens = []
    for sent in sentences:
        tokens = tokenize_word_level(sent)
        all_tokens.extend(tokens)
        
    log_message(f"Total tokens in evaluation set: {len(all_tokens)}")
    
    indices = encode_tokens(all_tokens, vocab)
    sequences = create_language_model_sequences(indices, seq_length)
    log_message(f"Created {len(sequences)} sequences")
    
    if not sequences:
        log_message("No sequences created. Data might be too short.")
        return {"loss": float('inf'), "perplexity": float('inf')}

    batches = create_batches(sequences, batch_size, shuffle=False)
    
    log_message("Starting evaluation...")
    start_time = time.time()
    
    total_loss = 0.0
    total_batches = len(batches)
    
    for i, batch in enumerate(batches):
        batch_loss = 0.0
        for input_seq, target_seq in batch:
            # We don't need gradients for evaluation, but the loss function returns them
            # Ideally we would have a separate forward_loss function without gradients, 
            # but for now we just ignore them.
            loss, _ = language_model_loss(input_seq, target_seq, model)
            batch_loss += loss
            
        total_loss += batch_loss / len(batch)
        
        if (i + 1) % 10 == 0:
            log_message(f"Processed {i + 1}/{total_batches} batches...")
            
    avg_loss = total_loss / total_batches
    perplexity = perplexity_from_loss(avg_loss)
    eval_time = time.time() - start_time
    
    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "time_seconds": eval_time
    }
    
    log_message("-" * 40)
    log_message(f"Evaluation Results:")
    log_message(f"Loss: {avg_loss:.4f}")
    log_message(f"Perplexity: {perplexity:.2f}")
    log_message(f"Time: {format_time(eval_time)}")
    log_message("-" * 40)
    
    return results
