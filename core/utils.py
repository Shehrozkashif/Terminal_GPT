from typing import Any
import pickle, time, math


def save_pickle(data: Any, filepath: str) -> None:
    """Save data using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def log_message(message: str, level: str = "INFO") -> None:
    """Print log message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def print_progress(current: int, total: int, prefix: str = "", 
                  bar_length: int = 50) -> None:
    """Print progress bar to terminal."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix} |{bar}| {percent*100:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)

def learning_rate_schedule(initial_lr: float, epoch: int, strategy: str = "exponential", decay_rate: float = 0.95) -> float:
    """
    Calculate learning rate for the current epoch.
    
    Args:
        initial_lr: Starting learning rate
        epoch: Current epoch (0-indexed)
        strategy: Decay strategy ('exponential' or 'step')
        decay_rate: Rate of decay
        
    Returns:
        New learning rate
    """
    if strategy == "exponential":
        return initial_lr * (decay_rate ** epoch)
    elif strategy == "step":
        # Decay every 5 epochs
        return initial_lr * (decay_rate ** (epoch // 5))
    else:
        return initial_lr
