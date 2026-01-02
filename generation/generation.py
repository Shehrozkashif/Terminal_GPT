import sys
from core.utils import load_pickle, log_message
from core.vocabulary import load_vocab, create_reverse_vocab, encode_tokens, decode_indices
from core.tokenizer import tokenize_word_level, detokenize
from models.language_model import generate_text


def generate_from_prompt(
    model_path: str,
    vocab_path: str,
    prompt: str = "",
    length: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
):
    """Generate text from prompt."""

    log_message("Loading model...")
    model = load_pickle(model_path)
    vocab = load_vocab(vocab_path)

                                    
    if prompt:
        prompt_tokens = tokenize_word_level(prompt)
        start_tokens = encode_tokens(prompt_tokens, vocab)
    else:
                   
        start_tokens = [2]

    log_message(f"Generating with prompt: '{prompt}'")
    log_message("-" * 60)

    generated_indices = generate_text(
        model,
        start_tokens,
        length,
        temperature,
        top_k,
    )

                             
    reverse_vocab = create_reverse_vocab(vocab)
    generated_tokens = decode_indices(
        generated_indices, reverse_vocab
    )
    generated_text = detokenize(generated_tokens)

    print("\n" + generated_text + "\n")
    log_message("Generation complete!")


def chat_loop(model_path: str, vocab_path: str):
    """Interactive chat loop."""

    log_message("Loading model...")
    model = load_pickle(model_path)
    vocab = load_vocab(vocab_path)
    reverse_vocab = create_reverse_vocab(vocab)

    conversation_history = []

    print("\n" + "=" * 60)
    print("MINI GPT CHATBOT")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

                                     
        tokens = tokenize_word_level(user_input)
        indices = encode_tokens(tokens, vocab)

        conversation_history.extend(indices)
        context = conversation_history[-100:]

                                     
        generated = generate_text(
            model,
            context,
            max_length=50,
            temperature=0.8,
            top_k=40,
        )

        response_indices = generated[len(context):]
        response_tokens = decode_indices(
            response_indices, reverse_vocab
        )
        response_text = detokenize(response_tokens)

        print(f"Bot: {response_text}\n")

        conversation_history.extend(response_indices)
