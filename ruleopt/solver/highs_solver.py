from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix, eye as speye, hstack as shstack
from ..utils import check_module_available
from .base import OptimizationSolver

HIGHS_AVAILABLE = check_module_available("highspy")


class HiGHSSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the HiGHS solver.

    The solver supports both dense and sparse matrix representations.
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
        """
        Parameters
        ----------
        coefficients : object
            An object containing the sparse matrix coefficients ('yvals', 'rows', 'cols'),
            and costs associated with each rule ('costs').
        k : float
            A scaling factor for the coefficients.
        ws0 : array-like, optional
            Initial weights for the optimization process. If provided, should have the same
            length as the number of rules. Otherwise, weights are initialized to ones.

        Returns
        -------
        ws : numpy.ndarray
            The optimized weights for each rule after the optimization process.
        betas : numpy.ndarray
            The betas values indicating constraint violations for the optimized solution.
        """
        ### LAZY IMPORT
        import highspy

        a_hat = csr_matrix(
            (
                coefficients.yvals,
                (coefficients.rows, coefficients.cols),
            ),
            dtype=np.float64,
        ) * ((k - 1.0) / k)

        if not self.use_sparse:
            a_hat = a_hat.toarray()

        costs = np.array(coefficients.costs, copy=False)

        unique_rows, adjusted_sample_weight, inverse_indices = self.group_contraints(
            a_hat, sample_weight
        )

        n, m = a_hat.shape
        num_vs = unique_rows.shape[0]
        total_vars = num_vs + m

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("solver", "ipm")

        inf = highspy.kHighsInf

        # Objective coefficients: [adjusted_sample_weight | costs * penalty * normalization_constant]
        obj = np.concatenate([
            adjusted_sample_weight,
            costs * self.penalty * normalization_constant,
        ])

        # Variable bounds: all >= 0, no upper bound
        col_lower = np.zeros(total_vars)
        col_upper = np.full(total_vars, inf)

        # Add columns (variables) without constraint entries
        h.addCols(
            total_vars,
            obj,
            col_lower,
            col_upper,
            0,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
        )

        # Build constraint matrix [I | unique_rows] in CSR format
        I_part = speye(num_vs, format="csr")
        if isinstance(unique_rows, np.ndarray):
            ur_csr = csr_matrix(unique_rows)
        else:
            ur_csr = unique_rows.tocsr()

        A = shstack([I_part, ur_csr], format="csr")

        # Row bounds: >= 1.0, no upper bound
        row_lower = np.ones(num_vs)
        row_upper = np.full(num_vs, inf)

        # Add rows (constraints) with sparse matrix
        h.addRows(
            num_vs,
            row_lower,
            row_upper,
            A.nnz,
            A.indptr.astype(np.int32),
            A.indices.astype(np.int32),
            A.data.astype(np.float64),
        )

        # Warm-start
        if ws0 is not None:
            col_value = np.zeros(total_vars)
            tempws = np.zeros(m)
            tempws[: len(ws0)] = ws0
            col_value[num_vs:] = tempws
            # Compute feasible vs values
            if isinstance(unique_rows, np.ndarray):
                vs_vals = np.maximum(1.0 - unique_rows @ tempws, 0.0)
            else:
                vs_vals = np.maximum(1.0 - unique_rows.dot(tempws), 0.0).A1
            col_value[:num_vs] = vs_vals
            h.setSolution(list(col_value), [])

        h.run()

        solution = h.getSolution()

        ws_X = np.array(solution.col_value[num_vs: num_vs + m], dtype=np.float64)
        duals_unique = np.array(solution.row_dual[:num_vs], dtype=np.float64)

        betas = self.fill_betas(
            n, duals_unique, inverse_indices.ravel(), sample_weight, rng
        )

        return ws_X, betas
