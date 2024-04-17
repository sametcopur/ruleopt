from typing import Tuple
from sklearn.utils import check_array
import numpy as np
from numpy.typing import ArrayLike
import importlib.util


def check_inputs(x: ArrayLike, y: ArrayLike = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates and preprocesses input data for the model.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        The training input samples. Internally, it will be converted to dtype=np.float32.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels) as integers

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The validated and preprocessed input matrix `x` and target vector `y`.
    """
    x = check_array(x, dtype=np.float32, force_all_finite="allow-nan")
    if y is not None:
        y = check_array(y, ensure_2d=False, dtype=np.intp)
        return x, y
    return x


def check_sample_weight(sample_weight: ArrayLike):
    return check_array(sample_weight, ensure_2d=False, dtype=np.float32)


def check_module_available(module_name):
    """
    Checks module is installed.
    """
    spec = importlib.util.find_spec(module_name)
    return spec is not None
