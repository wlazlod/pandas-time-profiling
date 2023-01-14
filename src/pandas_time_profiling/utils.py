"""Various utils functions
"""
from pathlib import Path
def get_project_root() -> Path:
    """Returns the path to the project root folder.
    Returns:
        The path to the project root folder.
    """
    return Path(__file__).parent
