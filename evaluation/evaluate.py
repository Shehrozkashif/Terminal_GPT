import time
from typing import Tuple, Dict
from core.dataset import load_text_file, create_language_model_sequences, create_batches, load_pickle, log_message
from core.tokenizer import tokenize_word_level, split_sentences
from core.vocabulary import encode_tokens, load_vocab
from models.language_model import language_model_forward, language_model_loss
from nn.losses import perplexity_from_loss
from nn.activations import softmax
from core.utils import format_time
from evaluation.metrics import calculate_accuracy, calculate_top_k_accuracy, calculate_precision_recall_f1

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
        Dictionary containing evaluation metrics (loss, perplexity, accuracy, f1, etc.)
    """
    log_message(f"Loading model from {model_path}...")
    model = load_pickle(model_path)
    
    log_message(f"Loading vocabulary from {vocab_path}...")
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    
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
    
    # Collect predictions and targets for metrics
    all_predictions = []
    all_targets = []
    all_logits = []
    
    for i, batch in enumerate(batches):
        batch_loss = 0.0
        for input_seq, target_seq in batch:
            # Forward pass
            outputs, _ = language_model_forward(input_seq, model)
            
            # Calculate loss
            loss, _ = language_model_loss(input_seq, target_seq, model)
            batch_loss += loss
            
            # Collect predictions for metrics
            for logits, target in zip(outputs, target_seq):
                probs = softmax(logits)
                prediction = logits.index(max(logits))  # Argmax
                
                all_predictions.append(prediction)
                all_targets.append(target)
                all_logits.append(logits)
            
        total_loss += batch_loss / len(batch)
        
        if (i + 1) % 10 == 0:
            log_message(f"Processed {i + 1}/{total_batches} batches...")
            
    avg_loss = total_loss / total_batches
    perplexity = perplexity_from_loss(avg_loss)
    eval_time = time.time() - start_time
    
    # Calculate additional metrics
    log_message("Calculating metrics...")
    accuracy = calculate_accuracy(all_predictions, all_targets)
    top5_accuracy = calculate_top_k_accuracy(all_logits, all_targets, k=5)
    top10_accuracy = calculate_top_k_accuracy(all_logits, all_targets, k=10)
    
    # Calculate precision, recall, F1 (this might take a while for large vocab)
    log_message("Calculating F1 score (this may take a moment)...")
    prf_metrics = calculate_precision_recall_f1(all_predictions, all_targets, vocab_size)
    
    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "precision": prf_metrics["precision"],
        "recall": prf_metrics["recall"],
        "f1_score": prf_metrics["f1"],
        "time_seconds": eval_time
    }
    
    # Display results
    log_message("=" * 50)
    log_message("EVALUATION RESULTS")
    log_message("=" * 50)
    log_message(f"Loss:              {avg_loss:.4f}")
    log_message(f"Perplexity:        {perplexity:.2f}")
    log_message("-" * 50)
    log_message(f"Accuracy:          {accuracy:.2f}%")
    log_message(f"Top-5 Accuracy:    {top5_accuracy:.2f}%")
    log_message(f"Top-10 Accuracy:   {top10_accuracy:.2f}%")
    log_message("-" * 50)
    log_message(f"Precision (macro): {prf_metrics['precision']:.2f}%")
    log_message(f"Recall (macro):    {prf_metrics['recall']:.2f}%")
    log_message(f"F1 Score (macro):  {prf_metrics['f1']:.2f}%")
    log_message("-" * 50)
    log_message(f"Evaluation Time:   {format_time(eval_time)}")
    log_message("=" * 50)
    
    return results
