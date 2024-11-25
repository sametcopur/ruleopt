from abc import ABC, abstractmethod
from typing import Any
import warnings
import numpy as np
from .solver_utils import fill_betas


class OptimizationSolver(ABC):
    """
    This abstract base class defines the interface for a generic solver.
    Implementations of this class must provide the `__call__` method,
    allowing the solver to be invoked as if it were a function.
    """

    def __init__(self) -> None:
        super().__init__()
        self.penalty: float | int
        self.use_sparse: bool
        self.solver_type: str | None
        self.lr: float | int | None
        self.constraint_cost: float | int | None
        self.weight_decay: float | int | None
        self.patience: int | None
        self.device: str | None
        self._check_params()

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Executes the solver using the provided arguments and keyword arguments.

        Parameters:
            *args (Any): Positional arguments required for solving the problem.
            **kwds (Any): Keyword arguments required for solving the problem.

        Returns:
            Any: The result of the solving process.
        """
        pass

    def group_contraints(self, a_hat: np.ndarray, sample_weight: np.ndarray):
        a_hat_view = np.ascontiguousarray(a_hat).view(
            np.dtype((np.void, a_hat.dtype.itemsize * a_hat.shape[1]))
        )

        # Find unique rows and inverse indices
        unique_a_hat_view, inverse_indices = np.unique(a_hat_view, return_inverse=True)

        # Recover the unique rows from the structured array
        unique_rows = unique_a_hat_view.view(a_hat.dtype).reshape(-1, a_hat.shape[1])

        # Compute the adjusted sample weights
        adjusted_sample_weight = np.bincount(
            inverse_indices.ravel(), weights=sample_weight
        )

        return unique_rows, adjusted_sample_weight, inverse_indices

    def fill_betas(
        self,
        n: int,
        duals_unique: np.ndarray,
        inverse_indices: np.ndarray,
        sample_weight: np.ndarray,
        rng: object,
    ):
        return fill_betas(n, duals_unique, inverse_indices, sample_weight, rng)

    def _check_params(self):
        if not hasattr(self, "penalty"):
            raise AttributeError("Subclasses must define 'penalty'")

        if not hasattr(self, "use_sparse"):
            raise AttributeError("Subclasses must define 'use_sparse'")

        if not isinstance(self.penalty, (float, int)) or self.penalty <= 0:
            raise TypeError("penalty must be a positive float.")

        if not isinstance(self.use_sparse, bool):
            raise TypeError(f"use_sparse must be True or False.")

        if hasattr(self, "solver_type"):
            valid_solvers = [
                "CLP",
                "PDLP",
                "GLOP",
                "GUROBI_LP",
                "CPLEX_LP",
                "XPRESS_LP",
                "GLPK_LP",
                "HiGHS",
            ]
            if (
                not isinstance(self.solver_type, str)
                or self.solver_type not in valid_solvers
            ):
                raise ValueError(f"solver_type must be one of {valid_solvers}.")

        if hasattr(self, "lr"):
            if not isinstance(self.lr, (float, int)) or self.lr <= 0:
                raise TypeError("lr must be a positive float.")

        if hasattr(self, "constraint_cost"):
            if (
                not isinstance(self.constraint_cost, (float, int))
                or self.constraint_cost <= 0
            ):
                raise TypeError("constraint_cost must be a positive float.")

        if hasattr(self, "weight_decay"):
            if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
                raise TypeError("weight_decay must be a non-negative float.")

        if hasattr(self, "patience"):
            if not isinstance(self.patience, int) or self.patience <= 0:
                raise TypeError("patience must be a positive integer.")

        if hasattr(self, "device"):
            valid_devices = ["cuda", "cpu"]
            if not isinstance(self.device, str) or self.device not in valid_devices:
                raise ValueError(f"solver_type must be one of {valid_devices}.")

        if self.use_sparse:
            warnings.warn(
                "A sparse data format is being used. If your dataset is not sufficiently "
                "large, using a sparse format could lead to performance issues.",
            )
