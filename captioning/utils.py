from .config import interactive
from pathlib import Path


def ask_overwrite(path: str) -> bool:
    """
    Args:
        path (str): Path to file

    Returns:
        bool: True if:
            1) Path does not exist (safe to create the file)
            2) user confirmed overwrite
    """
    if not Path(path).exists():
        return True

    # path exists
    if not interactive:
        return False

    while (
        response := input(f"{path} already present. Overwrite? [y/N]").lower()
    ) not in "yn":
        print(
            "Expecting one of 'YyNn'."
            f"For default press enter. You've typed: {response}"
        )
    return response == "y"
