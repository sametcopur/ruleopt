from abc import ABC, abstractmethod
from typing import Any
import warnings


class OptimizationSolver(ABC):
    """
    This abstract base class defines the interface for a generic solver.
    Implementations of this class must provide the `__call__` method, 
    allowing the solver to be invoked as if it were a function.
    """

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "penalty"):
            raise AttributeError("Subclasses must define 'penalty'")
        if not hasattr(self, "use_sparse"):
            raise AttributeError("Subclasses must define 'use_sparse'")

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

    def _check_params(self):
        if not isinstance(self.penalty, (float, int)) or self.penalty <= 0:
            raise TypeError("penalty must be a positive float.")

        if not isinstance(self.use_sparse, bool):
            raise TypeError(f"use_sparse must be True or False.")

        if hasattr(self, "solver_type"):
            valid_solvers = [
                "CLP",
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
            if not isinstance(self.constraint_cost, (float, int)) or self.constraint_cost <= 0:
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
