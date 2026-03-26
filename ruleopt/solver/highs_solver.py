from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import check_module_available
from .base import OptimizationSolver

HIGHS_AVAILABLE = check_module_available("highspy")


class HiGHSSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the HiGHS solver.

    Solves the dual LP directly:

    .. code-block:: text

        max  1^T beta
        s.t. U^T beta <= c
             0 <= beta <= s

    This avoids the identity block and v variables entirely.
    beta gives the dual values directly, ws is read from row_dual.
    """

    def __new__(cls, *args, **kwargs):
        if not HIGHS_AVAILABLE:
            raise ImportError(
                "HiGHS is required for this class but is not installed.",
                "Please install it with 'pip install highspy'",
            )
        instance = super(HiGHSSolver, cls).__new__(cls)
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
        self._h = None
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
        import highspy

        if self._h is None:
            self._h = highspy.Highs()
            self._h.setOptionValue("output_flag", False)
            self._h.setOptionValue("solver", "ipm")
            self._h.setOptionValue("run_crossover", "off")
            self._h.setOptionValue("presolve", "off")
            self._h.setOptionValue("threads", 0)
            self._h.setOptionValue("ipm_optimality_tolerance", 1e-4)

        h = self._h
        h.clearModel()

        # Build a_hat with scaled yvals in one shot
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
        num_unique = unique_rows_sp.shape[0]

        inf = highspy.kHighsInf

        # ── Dual LP: min -1^T β  s.t. U^T β <= c,  0 <= β <= s ──

        # Variables: β (num_unique)
        col_cost = -np.ones(num_unique, dtype=np.float64)
        col_lower = np.zeros(num_unique, dtype=np.float64)
        col_upper = adjusted_sample_weight.astype(np.float64)

        h.addCols(
            num_unique, col_cost, col_lower, col_upper,
            0,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

        # Constraints: U^T β <= c  (m rows)
        if not isinstance(unique_rows_sp, csr_matrix):
            unique_rows_sp = csr_matrix(unique_rows_sp)
        UT = unique_rows_sp.tocsc().T  # CSC.T yields CSR without sorting

        row_lower = np.full(m, -inf)
        row_upper = (costs * self.penalty * normalization_constant).astype(np.float64)

        h.addRows(
            m, row_lower, row_upper,
            UT.nnz,
            UT.indptr.astype(np.int32),
            UT.indices.astype(np.int32),
            UT.data,
        )

        h.run()

        solution = h.getSolution()

        # β directly from col_value
        duals_unique = np.array(solution.col_value[:num_unique], dtype=np.float64)

        # ws from row_dual of U^T β <= c (negated — HiGHS returns negative duals for <= constraints)
        row_duals = np.array(solution.row_dual[:m], dtype=np.float64)
        ws_X = np.maximum(-row_duals, 0.0)
        # Zero-clamp: IPM tolerance can leave near-zero residuals
        ws_X[ws_X < 1e-6] = 0.0

        betas = self.fill_betas(
            n, duals_unique, inverse_indices.ravel(), sample_weight, rng
        )

        return ws_X, betas
