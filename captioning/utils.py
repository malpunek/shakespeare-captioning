from .config import interactive
from pathlib import Path


def ask_overwrite(path: str) -> bool:
    """

    Args:
        path (str): Path to file

    Returns:
        bool: True if user wants to overwrite.
    """
    if interactive and Path(path).exists():
        while (
            response := input(f"{path} already present. Overwrite? [y/N]").lower()
        ) not in "yn":
            print(
                "Expecting one of 'YyNn'."
                f"For default press enter. You've typed: {response}"
            )
        if response != "y":
            return False
    return True
