from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier

from .base_sklearn import _RUGSKLEARN
from ...rule_cost import Gini
from ...utils import check_inputs
from ...solver import ORToolsSolver


class RUGClassifier(_RUGSKLEARN):
    """
    Rule Generation algorithm for multi-class classification. This algorithm aims at
    producing a compact and interpretable model by employing optimization-bsed rule learning.
    """

    def __init__(
        self,
        solver=ORToolsSolver(),
        rule_cost=Gini(),
        max_rmp_calls=10,
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
    ):
        """
        Parameters
        ----------
        solver : OptimizationSolver, default=ORToolsSolver()
            An instance of a derived class inherits from the 'Optimization Solver' base class.
            The solver is responsible for optimizing the rule set based on the cost function
            and constraints.

        rule_cost : RuleCost or int, default=Gini()
            Defines the cost of rules, either as a specific calculation method (RuleCost instance)
            or a fixed cost

        max_rmp_calls : int, default=10
            Maximum number of Restricted Master Problem (RMP) iterations allowed during fitting.

        class_weight: dict, "balanced" or None, default=None
            A dictionary mapping class labels to their respective weights, the string "balanced"
            to automatically adjust weights inversely proportional to class frequencies,
            or None for no weights. Used to adjust the model in favor of certain classes.

        threshold : float, default=1.0e-6
            The minimum weight threshold for including a rule in the final model.

        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducible results.

        criterion : {"gini", "entropy", "log_loss"}, default="gini"
            The function to measure the quality of a split. Supported criteria are
            "gini" for the Gini impurity and "log_loss" and "entropy" both for the

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed.

        monotonic_cst : array-like of int of shape (n_features), default=None
            Indicates the monotonicity constraint to enforce on each feature.
            - 1: monotonic increase
            - 0: no constraint
            - -1: monotonic decrease
        """
        self._validate_parameters(
            max_rmp_calls,
            criterion,
            splitter,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            max_features,
            max_leaf_nodes,
            min_impurity_decrease,
            ccp_alpha,
            monotonic_cst,
        )

        super().__init__(
            threshold=threshold,
            random_state=random_state,
            solver=solver,
            rule_cost=rule_cost,
            class_weight=class_weight,
        )

        self.max_rmp_calls = int(max_rmp_calls)

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

        self._temp_rules = []

    def _fit_decision_tree(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray
    ) -> DecisionTreeClassifier:
        """
        Fits a decision tree to the data, taking into account sample weights.

        Parameters
        ----------
        x : np.ndarray
            Feature matrix of the training data.
        y : np.ndarray
            Target vector of the training data.
        sample_weight : np.ndarray
            Array of weights for the samples.

        Returns
        -------
        DecisionTreeClassifier
            A decision tree classifier fitted to the weighted data.
        """
        dt = DecisionTreeClassifier(
            random_state=self._rng.integers(np.iinfo(np.int16).max),
            criterion=self.criterion,
            splitter=self.splitter,
            class_weight=self.class_weight,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            monotonic_cst=self.monotonic_cst,
        )

        if sample_weight is not None:
            dt.class_weight = None

        # Fit the decision tree to the data
        return dt.fit(x, y, sample_weight=sample_weight)

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        """
        Fits the RUGClassifier model to the training data using a rule generation approach.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to dtype=np.float32.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples get equal weights.

        Returns
        -------
        RUGClassifier
            The fitted model, ready for making predictions.
        """
        x, y = check_inputs(x, y)

        sample_weight = self._get_sample_weight(sample_weight, y)

        if self._is_fitted:
            self._cleanup()

        treeno = 0
        fit_tree = self._fit_decision_tree(x, y, sample_weight=None)
        self.decision_trees_[treeno] = fit_tree

        self._get_class_infos(y)
        vec_y = self._preprocess(y)
        self._get_matrix(x, y, vec_y, fit_tree, treeno)

        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        ws, betas = self.solver(
            coefficients=self.coefficients_,
            k=self.k_,
            sample_weight=sample_weight,
            normalization_constant=normalization_constant,
            rng=self._rng,
        )

        # Rule generation
        for _ in range(self.max_rmp_calls):
            if np.all(betas <= 1e-6):
                break

            treeno += 1
            fit_tree = self._fit_decision_tree(x, y, sample_weight=betas)
            self.decision_trees_[treeno] = fit_tree

            no_improvement = self._get_matrix(
                x, y, vec_y, fit_tree, treeno, betas, normalization_constant
            )

            if no_improvement:
                break

            ws, betas = self.solver(
                coefficients=self.coefficients_,
                k=self.k_,
                ws0=ws.copy(),
                normalization_constant=normalization_constant,
                sample_weight=sample_weight,
                rng=self._rng,
            )

        self._fill_rules(ws)
        self._is_fitted = True

        return self

    def _validate_parameters(
        self,
        max_rmp_calls,
        criterion,
        splitter,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        min_impurity_decrease,
        ccp_alpha,
        monotonic_cst,
    ):
        # max_rmp_calls check
        if not isinstance(max_rmp_calls, (float, int)):
            raise TypeError("max_rmp_calls must be an integer.")

        if max_rmp_calls < 0:
            raise ValueError("max_rmp_calls must be a non-negative integer.")

        # criterion check
        if criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError(
                "criterion must be one of 'gini', 'entropy', or 'log_loss'."
            )

        # splitter check
        if splitter not in {"best", "random"}:
            raise ValueError("splitter must be 'best' or 'random'.")

        # max_depth check
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None.")
        if isinstance(max_depth, int) and max_depth < 1:
            raise ValueError("max_depth must be greater than 0.")

        # min_samples_split check
        if not isinstance(min_samples_split, (int, float)) or min_samples_split < 2:
            raise ValueError(
                "min_samples_split must be an integer or float greater than or equal to 2."
            )

        # min_samples_leaf check
        if not isinstance(min_samples_leaf, (int, float)) or min_samples_leaf < 1:
            raise ValueError(
                "min_samples_leaf must be an integer or float greater than or equal to 1."
            )

        # min_weight_fraction_leaf check
        if not isinstance(min_weight_fraction_leaf, float) or not (
            0.0 <= min_weight_fraction_leaf <= 1.0
        ):
            raise ValueError(
                "min_weight_fraction_leaf must be a float between 0.0 and 1.0."
            )

        # max_features check
        if (
            max_features is not None
            and not isinstance(max_features, (int, float, str))
            or (isinstance(max_features, str) and max_features not in {"sqrt", "log2"})
        ):
            raise ValueError(
                "max_features must be an integer, float, 'sqrt', 'log2', or None."
            )

        # max_leaf_nodes check
        if max_leaf_nodes is not None and (
            not isinstance(max_leaf_nodes, int) or max_leaf_nodes < 1
        ):
            raise ValueError("max_leaf_nodes must be a positive integer or None.")

        # min_impurity_decrease check
        if not isinstance(min_impurity_decrease, float) or min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease must be a non-negative float.")

        # ccp_alpha check
        if not isinstance(ccp_alpha, float) or ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be a non-negative float.")

        # monotonic_cst check
        if monotonic_cst is not None and (
            not isinstance(monotonic_cst, (list, tuple))
            or not all(isinstance(item, int) for item in monotonic_cst)
        ):
            raise ValueError("monotonic_cst must be an array-like of integers or None.")
