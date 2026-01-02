from typing import Tuple
import time
from core.tokenizer import tokenize_word_level, split_sentences
from core.vocabulary import build_vocab_from_tokens, save_vocab, encode_tokens
from core.dataset import load_text_file, save_pickle, log_message, print_progress
from models.language_model import initialize_language_model, language_model_forward, language_model_loss
from nn.optimizer import sgd_step
from core.dataset import create_language_model_sequences, split_train_val, create_batches
from core.utils import format_time
from nn.losses import perplexity_from_loss
from core.dataset import create_language_model_sequences
from core.utils import learning_rate_schedule



def train_language_model(
    data_path: str,
    model_type: str = "lstm",
    hidden_size: int = 128,
    embedding_dim: int = 64,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    seq_length: int = 30,
) -> Tuple[dict, dict]:
    """Train language model."""

    log_message("Loading and preprocessing data...")

    text = load_text_file(data_path)
    sentences = split_sentences(text)

    all_tokens = []
    for sent in sentences:
        tokens = tokenize_word_level(sent)
        all_tokens.extend(tokens)

    log_message(f"Total tokens: {len(all_tokens)}")

    vocab = build_vocab_from_tokens(all_tokens, min_freq=2, max_vocab_size=5000)
    log_message(f"Vocabulary size: {len(vocab)}")

    indices = encode_tokens(all_tokens, vocab)
    sequences = create_language_model_sequences(indices, seq_length)
    log_message(f"Created {len(sequences)} sequences")

    train_data, val_data = split_train_val(sequences, val_split=0.1)

    log_message(f"Initializing {model_type.upper()} model...")
    model = initialize_language_model(len(vocab), embedding_dim, hidden_size, model_type)

    log_message("Starting training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        lr = learning_rate_schedule(learning_rate, epoch, strategy="exponential", decay_rate=0.95)

        batches = create_batches(train_data, batch_size, shuffle=True)

        total_loss = 0.0

        for batch_idx, batch in enumerate(batches):
            batch_loss = 0.0

            for input_seq, target_seq in batch:
                                        
                outputs, cache = language_model_forward(input_seq, model)

                                                    
                loss, gradients = language_model_loss(input_seq, target_seq, model)
                batch_loss += loss

                                            
                for param_name, param in model["params"].items():
                    if param_name in gradients["params"]:
                        grad = gradients["params"][param_name]
                        model["params"][param_name] = sgd_step(param, grad, lr)

            total_loss += batch_loss / len(batch)

            if (batch_idx + 1) % 5 == 0:
                print_progress(batch_idx + 1, len(batches), prefix=f"Epoch {epoch+1}/{num_epochs}")

        avg_loss = total_loss / len(batches)
        epoch_time = time.time() - epoch_start

                              
        val_loss = 0.0
        eval_samples = min(50, len(val_data))

        for input_seq, target_seq in val_data[:eval_samples]:
            outputs, cache = language_model_forward(input_seq, model)
            loss, gradients = language_model_loss(input_seq, target_seq, model)
            val_loss += loss

        val_loss /= eval_samples

        log_message(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Perplexity: {perplexity_from_loss(val_loss):.2f}, "
            f"Time: {format_time(epoch_time)}, "
            f"LR: {lr:.6f}"
        )

    total_time = time.time() - start_time
    log_message(f"Training complete! Total time: {format_time(total_time)}")

    save_pickle(model, "language_model.pkl")
    save_vocab(vocab, "vocab.txt")
    log_message("Model and vocabulary saved")

    return model, vocab
