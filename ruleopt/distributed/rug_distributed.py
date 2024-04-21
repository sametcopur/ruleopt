from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
import numpy as np
from typing import Tuple

from .ray_solver_gurobi import gurobi_parallel_solver
from ..rule_cost import Gini
from ..solver import GurobiSolver
from ..estimator import RUGClassifier as RUGBASE
from ..utils import check_inputs, check_module_available

GUROBI_AVAILABLE = check_module_available("gurobipy")


class RUGClassifier(RUGBASE):
    """
    Parallel RUGClassifier
    """
    def __new__(cls, *args, **kwargs):
        if not GUROBI_AVAILABLE:
            raise ImportError(
                "Gurobi is required for this class but is not installed.",
                "Please install it with 'pip install gurobipy'",
            )
        instance = super(RUGClassifier, cls).__new__(cls)
        return instance
    
    def __init__(
        self,
        parallel=False,
        par_learning_rate: float = 0.01,
        par_threshold: float = 0.001,
        par_total_iter: int = 1000,
        par_n_dist: int = 5,
        par_momentum: float = 0,
        rule_cost=Gini(),
        max_rmp_calls=20,
        threshold: float = 1.0e-6,
        random_state: int | None = None,
        class_weight: dict | str | None = None,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: int | float | str = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        monotonic_cst: ArrayLike | None = None,
    ) -> None:
        """
        Initialize the Parallel RUGClassifier class with the specified parameters.

        Parameters:
        - threshold: The threshold for selecting rules based on their weight.
        - random_state: The seed for the random number generator.
        - penalty_parameter: The penalty parameter for the optimization problem.
        - rule_cost: The cost function for rules. This could be a string indicating a predefined cost function.
        - max_rmp_calls: The maximum number of calls to the Restricted Master Problem solver.
        - tree_parameters: A dictionary containing the parameters for the decision trees in the ensemble.
                          Example: {'max_depth': 10, 'min_samples_split': 2}
        """
        self.parallel = parallel
        self.par_learning_rate = par_learning_rate
        self.par_momentum = par_momentum
        self.par_threshold = par_threshold
        self.par_total_iter = par_total_iter
        self.par_n_dist = par_n_dist

        super().__init__(
            solver=GurobiSolver(),
            rule_cost=rule_cost,
            max_rmp_calls=max_rmp_calls,
            threshold=threshold,
            random_state=random_state,
            class_weight=class_weight,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )

    def _parsol(
        self,
        ws0: np.ndarray = None,
        ws_par: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Parallel solution process for the entire dataset.

        Args:
            ws0 (np.ndarray): Initial weights for all distributions.
            ws_par (np.ndarray, optional): Global ws parameters. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
            The final ws, betas, and ws_par values for all distributions.
        """
        a_hat = csr_matrix(
            (
                self.coefficients_.yvals,
                (self.coefficients_.rows, self.coefficients_.cols),
            ),
            dtype=np.float64,
        ) * ((self.k_ - 1.0) / self.k_)

        a_hat = a_hat.toarray()
        row_splits = np.array_split(np.arange(a_hat.shape[0]), self.par_n_dist)
        dist_a_hat = [a_hat[split] for split in row_splits]
        m = a_hat.shape[1]

        del a_hat

        costs = np.array(self.coefficients.costs, copy=False)
        theta = np.ones(shape=(self.par_n_dist, m))
        theta_old = np.ones(shape=(self.par_n_dist, m))

        if ws_par is None:
            ws_par = np.ones(shape=(m,))
        else:
            temp_ws_par = np.ones(shape=(m,))
            temp_ws_par[: len(ws_par)] = ws_par
            ws_par = temp_ws_par

        for _ in range(self.par_total_iter):

            results = gurobi_parallel_solver(
                self.parallel,
                par_n_dist=self.par_n_dist,
                dist_a_hat=dist_a_hat,
                costs=costs,
                ws_par=ws_par,
                theta=theta,
                ws0=ws0,
                penalty=2,
                learning_rate=self.par_learning_rate,
            )

            ws, betas = zip(*results)

            # Convert lists to numpy arrays for faster calculations
            ws = np.array(ws)
            theta_diff = theta - theta_old

            # Compute loss
            # loss = np.mean(np.abs(ws[:-1] - ws[1:]))
            loss = np.max(np.mean(np.abs(ws - ws_par), axis=0))

            if loss < self.par_threshold:
                print("Iteration: ", _)
                print("Loss: ", loss)
                print()
                break

            ws_par = np.mean(ws + (theta / self.par_learning_rate), axis=0)
            theta += self.par_learning_rate * (ws - ws_par) + self.par_momentum * (
                theta_diff
            )

        return ws, np.concatenate(betas), ws_par

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the Parallel RUGClassifier to the input data.

        Parameters:
        ----------
        x : numpy.array
            Input data of shape (n_samples, n_features).

        y : numpy.array
            Target values of shape (n_samples,).

        Returns:
        -------
        RUGClassifier
            The fitted RUGClassifier model.
        """
        x, y = check_inputs(x, y)

        # If the model has been fitted before, clean it up
        if self._is_fitted:
            self._cleanup()

        # Initialize the tree number
        treeno = 0

        # Fit a decision tree to the data without sample weights
        fit_tree = self._fit_decision_tree(x, y, sample_weight=None)

        # Store the fitted decision tree
        self.decision_trees_[treeno] = fit_tree

        # Extract and set properties of the target variable
        self._get_class_infos(y)

        # Preprocess the target values
        vec_y = self._preprocess(y)

        # Calculate the coefficients and other parameters for the optimization problem
        self._get_matrix(x, y, vec_y, fit_tree, treeno)
        
        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        # Solve the optimization problem
        ws, betas, ws_par = self._parsol()

        # Column generation
        for _ in range(self.max_rmp_calls):
            # Increment the tree number
            treeno += 1

            # Fit a decision tree to the data using the dual variables as sample weights
            fit_tree = self._fit_decision_tree(x, y, sample_weight=betas)

            # Store the fitted decision tree
            self.decision_trees_[treeno] = fit_tree

            # Perform the PSPDT operation
            no_improvement = self._get_matrix(x, y, vec_y, fit_tree, treeno, betas, normalization_constant)

            # If there was no improvement, break the loop
            if no_improvement:
                break

            # Solve the optimization problem again with the new rules
            ws, betas, ws_par = self._parsol(ws.copy(), ws_par.copy())

        # Fill the decision rules based on the weights obtained from the optimization problem
        self._fill_rules(ws_par)

        # Mark the model as fitted
        self._is_fitted = True

        # Return the fitted model
        return self
