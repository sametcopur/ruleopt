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

    def __call__(self, coefficients, k, sample_weight, normalization_constant, *args, **kwargs) -> Any:
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
        from ortools.linear_solver.python import model_builder
        from pandas import Index

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

        costs = np.array(coefficients.costs, dtype=np.float32)

        model = model_builder.Model()
        solver = model_builder.Solver(self.solver_type)

        if not solver.solver_is_supported():
            raise ValueError(
                f"Support for {self.solver_type} not linked in, or the license ",
                "was not found.",
            )

        # Variables
        vs = model.new_num_var_series(
            name="vs", index=Index(np.arange(n)), lower_bounds=0
        )
        ws = model.new_num_var_series(
            name="ws", index=Index(np.arange(m)), lower_bounds=0
        )

        # Objective
        model.minimize(vs @ sample_weight + normalization_constant * self.penalty * costs @ ws)

        # Constraints
        model.add((a_hat @ ws + vs).map(lambda x: x >= 1))

        solver.solve(model)

        ws = solver.values(ws).values
        betas = solver.dual_values(model._get_linear_constraints()).values

        return ws, betas

    def _validate_parameters(self, solver_type, penalty_parameter):
        valid_solvers = [
            "CLP",
            "GLOP",
            "GUROBI_LP",
            "CPLEX_LP",
            "XPRESS_LP",
            "GLPK_LP",
            "HiGHS",
        ]
        if not isinstance(solver_type, str) or solver_type not in valid_solvers:
            raise ValueError(f"solver_type must be one of {valid_solvers}.")

        if not isinstance(penalty_parameter, (float, int)) or penalty_parameter <= 0:
            raise ValueError("penalty_parameter must be a positive float or integer.")
