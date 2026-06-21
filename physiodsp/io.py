import pathlib
from typing import Any
from scipy.io import loadmat, whosmat


def read_mat(filename: str | pathlib.Path) -> Any:
    """
    Reads a MATLAB .mat file and returns the first variable found in the file.

    Args:
        filename (str | pathlib.Path): Path to the .mat file.

    Returns:
        Any: The content of the first variable found in the .mat file.

    Raises:
        ValueError: If the file extension is not .mat or if no variables are found.
        FileNotFoundError: If the file does not exist.
    """
    path = pathlib.Path(filename)
    if path.suffix != '.mat':
        raise ValueError(f"Unsupported file extension: {path.suffix}. Only .mat files are supported.")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    mat_vars = whosmat(str(path))
    if not mat_vars:
        raise ValueError(f"No variables found in the .mat file: {path}")

    # The first element of the first tuple is the name of the matlab data structure
    var_name = mat_vars[0][0]
    mat = loadmat(str(path), simplify_cells=True)
    return mat[var_name]
