from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import check_module_available
from .base import OptimizationSolver

CPLEX_AVAILABLE = check_module_available("docplex")


class CPLEXSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the CPLEX solver.

    The solver supports both dense and sparse matrix representations.
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
            The optimal weights for each rule.
        betas : numpy.ndarray
            The optimal dual solution.
        """
        ### LAZY IMPORT
        from docplex.mp.model import Model

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

        n, m = a_hat.shape
        # Primal Model
        modprimal = Model("RUXG Primal")

        unique_rows, adjusted_sample_weight, inverse_indices = self.group_contraints(
            a_hat, sample_weight
        )

        # Variables
        vs = modprimal.continuous_var_list(unique_rows.shape[0], name="vs")
        ws = modprimal.continuous_var_list(m, name="ws")

        # Set initial values
        initial_values = []

        if ws0 is not None:
            initial_values += [(ws[i], ws0[i]) for i in range(len(ws0))]

        # Assign initial solution
        modprimal.start = initial_values

        # Objective
        modprimal.minimize(
            modprimal.sum(vs * adjusted_sample_weight)
            + modprimal.scal_prod(ws, costs * self.penalty * normalization_constant)
        )
        # Constraints
        for i in range(unique_rows.shape[0]):
            modprimal.add_constraint(
                modprimal.sum(unique_rows[i, j] * ws[j] for j in range(m)) + vs[i]
                >= 1.0
            )

        modprimal.solve()

        duals_unique = np.array(
            [c.dual_value for c in modprimal.iter_constraints()], dtype=np.float64
        )

        betas = self.fill_betas(
            n, duals_unique, inverse_indices.ravel(), sample_weight, rng
        )
        ws = np.array([v.solution_value for v in ws], dtype=np.float64)

        return ws, betas
