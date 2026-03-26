from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import check_module_available
from .base import OptimizationSolver

GUROBI_AVAILABLE = check_module_available("gurobipy")


class GurobiSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the Gurobi solver.

    Solves the dual LP directly:

    .. code-block:: text

        max  1^T beta
        s.t. U^T beta <= c
             0 <= beta <= s
    """

    def __new__(cls, *args, **kwargs):
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi is required for this class but is not installed.",
                "Please install it with 'pip install gurobipy'",
            )
        instance = super(GurobiSolver, cls).__new__(cls)
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
        from gurobipy import Model, GRB

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

        unique_rows_sp, adjusted_sample_weight, inverse_indices = self.group_contraints(a_hat, sample_weight)

        if not isinstance(unique_rows_sp, csr_matrix):
            unique_rows_sp = csr_matrix(unique_rows_sp)

        num_unique = unique_rows_sp.shape[0]
        UT = unique_rows_sp.T.tocsr()

        # Dual LP: max 1^T β  s.t. U^T β <= c,  0 <= β <= s
        moddual = Model("RUG Dual")
        moddual.setParam("OutputFlag", False)
        moddual.setParam("Method", 3)
        moddual.setParam("Crossover", 0)
        moddual.setParam("Presolve", 0)
        moddual.setParam("BarConvTol", 1e-4)

        # Variables: β (num_unique), bounded [0, s]
        beta = moddual.addMVar(
            shape=num_unique,
            lb=0.0,
            ub=adjusted_sample_weight.astype(np.float64),
            name="beta",
        )

        # Objective: maximize 1^T β
        moddual.setObjective(np.ones(num_unique) @ beta, GRB.MAXIMIZE)

        # Constraints: U^T β <= c
        c_rhs = (costs * self.penalty * normalization_constant).astype(np.float64)
        moddual.addConstr(UT @ beta <= c_rhs, name="dual_constraints")

        moddual.optimize()

        # β directly from solution
        duals_unique = beta.X

        # ws from dual of U^T β <= c (Pi)
        ws_X = np.maximum(np.array(moddual.getAttr(GRB.Attr.Pi)[:m]), 0.0)
        ws_X[ws_X < 1e-6] = 0.0

        betas = self.fill_betas(n, duals_unique, inverse_indices.ravel(), sample_weight, rng)

        moddual.dispose()

        return ws_X, betas
