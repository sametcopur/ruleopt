from typing import Any
from scipy.sparse import csr_matrix
import numpy as np
from ..utils import check_module_available
from .base import OptimizationSolver

ORTOOLS_AVAILABLE = check_module_available("ortools")


class ORToolsSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the Google's OR-Tools solver.

    The solver supports both dense and sparse matrix representations.

    This solver can handle large-scale linear programming problems by interfacing with
    various backend solvers such as CLP, GLOP, and proprietary solvers like Gurobi and CPLEX.
    """

    def __new__(cls, *args, **kwargs):
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools is required for this class but is not installed.",
                "Please install it with 'pip install ortools'",
            )
        instance = super(ORToolsSolver, cls).__new__(cls)
        return instance

    def __init__(
        self,
        penalty: float = 1.0,
        solver_type: str = "GLOP",
        use_sparse: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        penalty : float, default=1.0
            Penalty parameter for the cost in the objective function.

        solver_type : {"CLP", "GLOP", "GUROBI_LP", "CPLEX_LP", "XPRESS_LP", "GLPK_LP", "HiGHS"}, default="GLOP"
            The type of Linear Programming solver to use.

        use_sparse : bool, default=False
            Determines whether to use a sparse matrix representation for the optimization
            problem. Using sparse matrices can significantly reduce memory usage and improve
            performance for large-scale problems with many zeros in the data.
        """
        self.solver_type = solver_type
        self.penalty = penalty
        self.use_sparse = use_sparse
        super().__init__()

    def __call__(
        self,
        coefficients,
        k,
        sample_weight,
        normalization_constant,
        rng,
        *args,
        **kwargs,
    ) -> Any:
        """
        Solves a linear optimization problem with the given coefficients and penalty.

        Parameters
        ----------
        coefficients : object
            An object containing the sparse matrix coefficients ('yvals', 'rows', 'cols'),
            and costs associated with each rule ('costs').
        k : float
            A scaling factor for the coefficients.

        Returns
        -------
        ws : numpy.ndarray
            The optimized weights for each rule after the optimization process.
        betas : numpy.ndarray
            The betas values indicating constraint violations for the optimized solution.

        Raises
        ------
        ValueError
            If the specified solver type is not supported or not linked correctly.
        """
        ### LAZY IMPORT
        from .solver_utils import solve_ortools

        a_hat = csr_matrix(
            (
                coefficients.yvals,
                (coefficients.rows, coefficients.cols),
            ),
            dtype=np.float64,
        ) * ((k - 1.0) / k)

        if not self.use_sparse:
            a_hat = a_hat.toarray()

        n, m = a_hat.shape

        unique_rows, adjusted_sample_weight, inverse_indices = self.group_contraints(
            a_hat, sample_weight
        )

        ws, duals_unique = solve_ortools(
            unique_rows,
            adjusted_sample_weight,
            normalization_constant,
            self.penalty,
            self.solver_type,
            coefficients.costs,
        )

        betas = self.fill_betas(
            n, duals_unique, inverse_indices.ravel(), sample_weight, rng
        )

        return ws, betas
