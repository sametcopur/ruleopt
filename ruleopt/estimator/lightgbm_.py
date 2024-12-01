from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

from .base import _RUGBASE
from ..aux_classes import Rule
from ..rule_cost import Gini
from ..utils import check_module_available, check_inputs
from ..solver import ORToolsSolver

LIGHTGBM_AVAILABLE = check_module_available("lightgbm")


class RUXLGBMClassifier(_RUGBASE):
    """
    A classifier that extracts and optimizes decision rules from a trained
    LightGBM ensemble model to create a compact and interpretable model.
    This process involves translating the ensemble's trees into a set of rules and
    using optimization to balance model accuracy and interpretability. The complexity
    of the resulting rule-based model is  controlled through a penalty parameter.
    """

    def __new__(cls, *args, **kwargs):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM is required for this class but is not installed.",
                "Please install it with 'pip install lightgbm'",
            )
        instance = super(RUXLGBMClassifier, cls).__new__(cls)
        return instance

    def __init__(
        self,
        trained_ensemble,
        *,
        solver=ORToolsSolver(),
        rule_cost=Gini(),
        class_weight: dict | str | None = None,
        threshold: float = 1.0e-6,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        trained_ensemble : lightgbm.LGBMClassifier or lightgbm.Booster
            The trained LightGBM ensemble model from which the rules will be extracted.

        solver : OptimizationSolver, default=ORToolsSolver()
            An instance of a derived class inherits from the 'Optimization Solver' base class.
            The solver is responsible for optimizing the rule set based on the cost function
            and constraints.

        rule_cost : RuleCost or int, default=Gini()
            Defines the cost of rules, either as a specific calculation method (RuleCost instance)
            or a fixed cost

        class_weight: dict, "balanced" or None, default=None
            A dictionary mapping class labels to their respective weights, the string "balanced"
            to automatically adjust weights inversely proportional to class frequencies,
            or None for no weights. Used to adjust the model in favor of certain classes.

        threshold : float, default=1.0e-6
            The minimum weight threshold for including a rule in the final model.

        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducible results.
        """
        ### LAZY INIT
        from lightgbm import Booster

        if not isinstance(trained_ensemble, Booster):
            if hasattr(trained_ensemble, "booster_"):
                if not isinstance(trained_ensemble.booster_, Booster):
                    raise TypeError("trained_ensemble is not an instance of LightGBM.")
                else:
                    self.trained_ensemble = trained_ensemble.booster_
            else:
                raise TypeError("trained_ensemble is not an instance of LightGBM")
        else:
            self.trained_ensemble = trained_ensemble

        super().__init__(
            threshold=threshold,
            random_state=random_state,
            rule_cost=rule_cost,
            solver=solver,
            class_weight=class_weight,
        )

    def _find_leaf_index(self, fit_tree: dict, leaf_index: int, path=None) -> list:
        """
        Recursively finds the path from the root to the specified leaf node in a LightGBM tree.

        Parameters
        ----------
        fit_tree : dict
            The tree dictionary from a LightGBM model.
        leaf_index : int
            The target leaf node's ID.
        path : list, optional
            The path taken to reach the current node, used in recursive calls. Defaults to None.

        Returns
        -------
        list
            The path from the root to the leaf node, represented as a list of node IDs.
        """
        if path is None:
            path = []
        if isinstance(fit_tree, dict):
            # If the current fit_tree is the leaf node we're looking for
            if "leaf_index" in fit_tree and fit_tree["leaf_index"] == leaf_index:
                return path
            # Recursively look in the items of the fit_tree
            for key, value in fit_tree.items():
                result = self._find_leaf_index(value, leaf_index, path + [key])
                if result:
                    return result

    def _get_rule(self, fit_tree: dict, nodeid: int) -> Rule:
        """
        Constructs a rule corresponding to the path leading to a specific leaf node in
        a LightGBM tree.

        Parameters
        ----------
        fit_tree : dict
            The tree structure extracted from a LightGBM model, typically in JSON format.
        nodeid : int
            The unique identifier of the leaf node for which to construct the rule.

        Returns
        -------
        Rule
            An object representing the decision rule leading to the specified leaf node.
        """
        # Get the path to the leaf node
        leaf_path = self._find_leaf_index(fit_tree, nodeid)

        # Initialize the rule
        return_rule = Rule()

        # child variable is used to track the last traversed node
        child = None

        # While there are still nodes in the path to the leaf node
        while leaf_path:
            fit_tree_ = fit_tree.copy()
            for path in leaf_path:
                fit_tree_ = fit_tree_[path]

            # If the current node is a split node, add a clause to the rule
            if "split_feature" in fit_tree_.keys():
                # Check which child node we're coming from
                is_left = child == "left_child"
                # Get the threshold for the split
                threshold = fit_tree_["threshold"]
                # Get the default path in case of missing values
                missing = (is_left and fit_tree_["default_left"]) or (
                    child == "right_child" and not fit_tree_["default_left"]
                )

                feature = fit_tree_["split_feature"]
                ub = threshold if is_left else np.inf
                lb = -np.inf if is_left else threshold
                na = missing

                return_rule.add_clause(feature, ub, lb, na)

            # Move to the next node in the path
            child = leaf_path.pop()

        return return_rule

    def _get_matrix(
        self,
        y: np.ndarray,
        vec_y: np.ndarray,
        fit_tree: dict,
        treeno: int,
        y_rules: np.ndarray,
    ):
        """
        Populates the matrices for the optimization problem based on a single LightGBM tree.

        Parameters
        ----------
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector, suitable for the optimization problem.
        fit_tree : dict
            A single decision tree's structure from LightGBM, in dictionary form.
        treeno : int
            The index of the current tree within the ensemble.
        y_rules : np.ndarray
            The array of leaf indices for each sample in the training data, determined by
            the current tree.
        """
        # If the coefficients matrix is empty, start from the first column
        if self.coefficients_.cols.shape[0] == 0:
            col = 0
        else:
            # Otherwise, start from the next available column
            col = np.max(self.coefficients_.cols) + 1

        # Get the leaf node for each sample in x
        y_rules = y_rules[:, treeno]

        # Iterate over unique leaf nodes
        for leafno in np.unique(y_rules):
            # Get the samples in the leaf
            covers = np.where(y_rules == leafno)[0]
            leaf_y_vals = y[covers]  # y values of the samples in the leaf

            # Get unique labels in the leaf and their counts
            unique_labels, counts = np.unique(leaf_y_vals, return_counts=True)

            # Identify the majority class in the leaf
            label = unique_labels[np.argmax(counts)]

            # Create a vector for this label
            label_vector = np.full((self.k_,), -1 / (self.k_ - 1))
            label_vector[label] = 1

            # Calculate fill_ahat, which will be used to update yvals in the coefficients matrix
            fill_ahat = np.dot(vec_y[covers, :], label_vector)

            # Update the coefficients matrix with the new information
            self.coefficients_.rows = np.concatenate((self.coefficients_.rows, covers))
            self.coefficients_.cols = np.concatenate(
                (self.coefficients_.cols, [col] * covers.shape[0])
            )
            self.coefficients_.yvals = np.concatenate(
                (self.coefficients_.yvals, np.full(covers.shape[0], fill_ahat))
            )

            cost = self._get_rule_cost(
                temp_rule=self._get_rule(fit_tree, leafno),
                covers=covers,
                counts=counts,
                y=y,
            )

            # Append the cost to the costs in the coefficients matrix
            self.coefficients_.costs = np.concatenate(
                (self.coefficients_.costs, [cost])
            )

            # Calculate the distribution of the samples in the leaf across the classess
            sdist = counts
            self.rule_info_[col] = (treeno, leafno, label, sdist)
            col += 1

    def _get_matrices(self, x: np.ndarray, y: np.ndarray, vec_y: np.ndarray):
        """
        Generates the coefficient matrices for the optimization problem from all
        trees in the LightGBM ensemble.

        Parameters
        ----------
        x : np.ndarray
            The feature matrix of the training data.
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector.
        """
        y_rules = self.trained_ensemble.predict(x, pred_leaf=True).astype(np.intp)
        for treeno, fit_tree in enumerate(self.decision_trees_.values()):
            self._get_matrix(y, vec_y, fit_tree, treeno, y_rules)

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        """
        Fits the RUXLGBMClassifier to the training data, optimizing the
        extracted rules for a balance between accuracy and interpretability.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to dtype=np.float32.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples equal weights.

        Returns
        -------
        RUXLGBMClassifier
            The fitted model, ready for making predictions.
        """

        x, y = check_inputs(x, y)

        sample_weight = self._get_sample_weight(sample_weight, y)

        # If the model has been fitted before, clean it up
        if self._is_fitted:
            self._cleanup()

        # Fills the fitted decision trees.
        tree_infos = self.trained_ensemble.dump_model()["tree_info"]
        for treeno, fit_tree in enumerate(tree_infos):
            self.decision_trees_[treeno] = fit_tree

        # Extract and set properties of the target variable
        self._get_class_infos(y)

        # Preprocess the target values
        vec_y = self._preprocess(y)

        # Calculate the coefficients and other parameters for the optimization problem
        self._get_matrices(x=x, y=y, vec_y=vec_y)

        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        # Solve the optimization problem again with the new rules
        ws, *_ = self.solver(
            coefficients=self.coefficients_,
            k=self.k_,
            sample_weight=sample_weight,
            normalization_constant=normalization_constant,
            rng=self._rng
        )

        # Fill the decision rules based on the weights obtained from the optimization problem
        self._fill_rules(ws)

        # Mark the model as fitted
        self._is_fitted = True

        # Return the fitted model
        return self
