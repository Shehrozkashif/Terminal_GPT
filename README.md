# Terminal_GPT

Terminal_GPT is a pure Python NLP framework designed to demonstrate the fundamentals of language modeling, including training, generation, and evaluation, directly from the terminal.

## Features

- **Train**: Train an LSTM-based language model on any text dataset.
- **Evaluate**: Evaluate the model's performance (Loss and Perplexity) on a test dataset.
- **Generate**: Generate text using the trained model with customizable temperature and top-k sampling.
- **Chat**: Interact with the model in a conversational loop.

## Installation

No external dependencies are strictly required for the core logic, but `pyparsing` is used in some utility functions.

```bash
pip install pyparsing
```

## Usage

### 1. Train a Model

Train a new language model using a text file.

```bash
python main.py train --data data/data.txt --epochs 10 --batch 16 --hidden 128
```

**Arguments:**
- `--data`: Path to training data (required)
- `--model`: Model type (default: `lstm`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.01)
- `--hidden`: Hidden layer size (default: 128)
- `--embed`: Embedding dimension (default: 64)
- `--seqlen`: Sequence length (default: 30)

**Example:**
```bash
python main.py train --data data/data.txt --epochs 20 --batch 32 --lr 0.005
```

---

### 2. Evaluate a Model

Evaluate the performance of a trained model on a dataset. This calculates comprehensive metrics including **Loss**, **Perplexity**, **Accuracy**, **Top-K Accuracy**, **Precision**, **Recall**, and **F1 Score**.

```bash
python main.py evaluate --data data/test_data.txt --model language_model.pkl
```

**Arguments:**
- `--data`: Path to evaluation data (required)
- `--model`: Path to trained model file (default: `language_model.pkl`)
- `--vocab`: Path to vocabulary file (default: `vocab.txt`)
- `--batch`: Batch size for evaluation (default: 16)
- `--seqlen`: Sequence length (default: 30)

**Example:**
```bash
python main.py evaluate --data data/validation.txt --model language_model.pkl --batch 32
```

**Output:**
```
==================================================
EVALUATION RESULTS
==================================================
Loss:              3.4759
Perplexity:        32.33
--------------------------------------------------
Accuracy:          42.65%
Top-5 Accuracy:    51.85%
Top-10 Accuracy:   60.26%
--------------------------------------------------
Precision (macro): 0.59%
Recall (macro):    1.39%
F1 Score (macro):  0.83%
--------------------------------------------------
Evaluation Time:   47s
==================================================
```

**Metrics Explained:**
- **Loss**: Cross-entropy loss - lower is better (measures prediction accuracy)
- **Perplexity**: exp(loss) - lower is better (measures how "surprised" the model is)
- **Accuracy**: Percentage of correctly predicted next tokens
- **Top-K Accuracy**: Percentage where correct token is in top-K predictions
- **Precision**: Macro-averaged precision across all vocabulary tokens
- **Recall**: Macro-averaged recall across all vocabulary tokens
- **F1 Score**: Harmonic mean of precision and recall

---

### 3. Generate Text

Generate text from a prompt.

```bash
python main.py generate --prompt "The future of AI" --length 50
```

**Arguments:**
- `--model`: Path to trained model file (default: `language_model.pkl`)
- `--vocab`: Path to vocabulary file (default: `vocab.txt`)
- `--prompt`: Starting text (default: empty string)
- `--length`: Number of tokens to generate (default: 100)
- `--temp`: Sampling temperature - higher = more random (default: 0.8)
- `--topk`: Top-k sampling parameter (default: 40)

**Example:**
```bash
python main.py generate --prompt "Once upon a time" --length 100 --temp 1.0
```

---

### 4. Chat

Start an interactive chat session with the model.

```bash
python main.py chat
```

**Arguments:**
- `--model`: Path to trained model file (default: `language_model.pkl`)
- `--vocab`: Path to vocabulary file (default: `vocab.txt`)

Type your message and press Enter. Type `quit` to exit.

---

## Project Structure

```
Terminal_GPT/
├── core/           # Basic utilities (tokenizer, vocabulary, dataset)
├── models/         # Model architectures (LSTM, Language Model wrapper)
├── nn/             # Neural network primitives (activations, losses, optimizer)
├── train/          # Training logic
├── evaluation/     # Model evaluation logic
├── generation/     # Text generation logic
├── data/           # Training and evaluation datasets
└── main.py         # CLI entry point
```

## Workflow Example

```bash
# 1. Train a model
python main.py train --data data/data.txt --epochs 15

# 2. Evaluate the model
python main.py evaluate --data data/data.txt

# 3. Generate text
python main.py generate --prompt "Hello world" --length 50

# 4. Chat with the model
python main.py chat
```