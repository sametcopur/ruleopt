from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix
from ..utils import check_module_available
from .base import OptimizationSolver

GUROBI_AVAILABLE = check_module_available("gurobipy")


class GurobiSolver(OptimizationSolver):
    """
    A solver wrapper class for linear optimization using the Gurobi solver.

    The solver supports both dense and sparse matrix representations.
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
        from gurobipy import Model, GRB

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
        
        unique_rows, adjusted_sample_weight, inverse_indices = self.group_contraints(a_hat, sample_weight)

        n, m = a_hat.shape

        modprimal = Model("RUG Primal")
        modprimal.setParam("OutputFlag", False)
        # Variables
        vs = modprimal.addMVar(shape=int(unique_rows.shape[0]), name="vs")
        ws = modprimal.addMVar(shape=int(m), name="ws")

        if ws0 is not None:
            tempws = np.zeros(m)
            tempws[: len(ws0)] = ws0
            ws.setAttr("Start", tempws)
            modprimal.update()

        # Objective
        modprimal.setObjective(
            adjusted_sample_weight @ vs + (costs * self.penalty * normalization_constant) @ ws,
            GRB.MINIMIZE,
        )
        # Constraints
        modprimal.addConstr(unique_rows @ ws + vs >= 1.0, name="a_hat Constraints")

        modprimal.optimize()

        duals_unique = np.array(modprimal.getAttr(GRB.Attr.Pi)[:n])

        betas = self.fill_betas(n, duals_unique, inverse_indices.ravel(), sample_weight, rng)
        
        return ws.X, betas
