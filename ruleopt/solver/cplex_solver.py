from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import check_module_available
from .base import OptimizationSolver

CPLEX_AVAILABLE = check_module_available("docplex")


class CPLEXSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the CPLEX solver.

    Solves the dual LP directly:

    .. code-block:: text

        max  1^T beta
        s.t. U^T beta <= c
             0 <= beta <= s
    """

    def __new__(cls, *args, **kwargs):
        if not CPLEX_AVAILABLE:
            raise ImportError(
                "CPLEX is required for this class but is not installed.",
                "Please install it with 'pip install docplex cplex'",
            )
        instance = super(CPLEXSolver, cls).__new__(cls)
        return instance

    def __init__(
        self,
        penalty: float = 1.0,
        use_sparse: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        penalty : float, default=1.0
            Penalty parameter for the cost in the objective function.
        use_sparse : bool, default=False
            Determines whether to use a sparse matrix representation for the optimization
            problem. Using sparse matrices can significantly reduce memory usage and improve
            performance for large-scale problems with many zeros in the data.
        """
        self.penalty = penalty
        self.use_sparse = use_sparse
        super().__init__()

    def __call__(
        self,
        coefficients,
        k: int,
        sample_weight,
        normalization_constant,
        rng,
        ws0: np.ndarray = None,
        *args,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ### LAZY IMPORT
        from docplex.mp.model import Model

        scale = (k - 1.0) / k
        a_hat = csr_matrix(
            (
                np.asarray(coefficients.yvals, dtype=np.float64) * scale,
                (coefficients.rows, coefficients.cols),
            ),
            dtype=np.float64,
        )

        n, m = a_hat.shape
        costs = np.array(coefficients.costs, copy=False)

        unique_rows_sp, adjusted_sample_weight, inverse_indices = self.group_contraints(
            a_hat, sample_weight
        )

        if not isinstance(unique_rows_sp, csr_matrix):
            unique_rows_sp = csr_matrix(unique_rows_sp)

        num_unique = unique_rows_sp.shape[0]
        UT = unique_rows_sp.T.toarray()  # docplex needs dense indexing

        c_rhs = (costs * self.penalty * normalization_constant).astype(np.float64)

        # Dual LP: max 1^T β  s.t. U^T β <= c,  0 <= β <= s
        moddual = Model("RUG Dual")

        # Variables: β bounded [0, s]
        beta = [
            moddual.continuous_var(lb=0.0, ub=float(adjusted_sample_weight[i]), name=f"b{i}")
            for i in range(num_unique)
        ]

        # Objective: maximize sum(β)
        moddual.maximize(moddual.sum(beta))

        # Constraints: U^T β <= c  (one per rule)
        for j in range(m):
            moddual.add_constraint(
                moddual.sum(UT[j, i] * beta[i] for i in range(num_unique) if UT[j, i] != 0)
                <= c_rhs[j]
            )

        moddual.solve()

        # β directly from solution
        duals_unique = np.array([v.solution_value for v in beta], dtype=np.float64)

        # ws from dual of U^T β <= c
        ws_X = np.array(
            [c.dual_value for c in moddual.iter_constraints()], dtype=np.float64
        )
        ws_X = np.maximum(ws_X, 0.0)
        ws_X[ws_X < 1e-6] = 0.0

        betas = self.fill_betas(
            n, duals_unique, inverse_indices.ravel(), sample_weight, rng
        )

        return ws_X, betas
