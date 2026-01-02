import argparse
from train.training import train_language_model
from evaluation.evaluate import evaluate_model
from generation.generation import generate_from_prompt, chat_loop


def main():
    """Main CLI function."""

    parser = argparse.ArgumentParser(
        description="Terminal GPT - Pure Python NLP Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py train --data data.txt

  Generate text:
    python main.py generate --prompt "hello world"

  Chat interactively:
    python main.py chat
"""
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Command to run"
    )

                             
    train_parser = subparsers.add_parser(
        "train", help="Train a language model"
    )
    train_parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training data"
    )
    train_parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["lstm"], help="Model type"
    )
    train_parser.add_argument(
        "--hidden", type=int, default=128,
        help="Hidden size"
    )
    train_parser.add_argument(
        "--embed", type=int, default=64,
        help="Embedding dimension"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--seqlen", type=int, default=30,
        help="Sequence length"
    )


    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a language model"
    )
    eval_parser.add_argument(
        "--data", type=str, required=True,
        help="Path to evaluation data"
    )
    eval_parser.add_argument(
        "--model", type=str, default="language_model.pkl",
        help="Model path"
    )
    eval_parser.add_argument(
        "--vocab", type=str, default="vocab.txt",
        help="Vocabulary path"
    )
    eval_parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size"
    )
    eval_parser.add_argument(
        "--seqlen", type=int, default=30,
        help="Sequence length"
    )

                                
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text"
    )
    gen_parser.add_argument(
        "--model", type=str, default="language_model.pkl",
        help="Model path"
    )
    gen_parser.add_argument(
        "--vocab", type=str, default="vocab.txt",
        help="Vocabulary path"
    )
    gen_parser.add_argument(
        "--prompt", type=str, default="",
        help="Starting prompt"
    )
    gen_parser.add_argument(
        "--length", type=int, default=100,
        help="Generation length"
    )
    gen_parser.add_argument(
        "--temp", type=float, default=0.8,
        help="Temperature"
    )
    gen_parser.add_argument(
        "--topk", type=int, default=40,
        help="Top-k sampling"
    )

                            
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat"
    )
    chat_parser.add_argument(
        "--model", type=str, default="language_model.pkl",
        help="Model path"
    )
    chat_parser.add_argument(
        "--vocab", type=str, default="vocab.txt",
        help="Vocabulary path"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_language_model(
            args.data,
            args.model,
            args.hidden,
            args.embed,
            args.epochs,
            args.batch,
            args.lr,
            args.seqlen,
        )

    elif args.command == "evaluate":
        evaluate_model(
            args.model,
            args.data,
            args.vocab,
            args.batch,
            args.seqlen,
        )

    elif args.command == "generate":
        generate_from_prompt(
            args.model,
            args.vocab,
            args.prompt,
            args.length,
            args.temp,
            args.topk,
        )

    elif args.command == "chat":
        chat_loop(args.model, args.vocab)

    else:
        parser.print_help()



if __name__ == "__main__":
    main()
