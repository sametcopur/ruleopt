from __future__ import annotations
import re
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .base import _RUGBASE
from ..aux_classes import Rule
from ..rule_cost import Gini
from ..utils import check_module_available, check_inputs
from ..solver import ORToolsSolver


XGBOOST_AVAILABLE = check_module_available("xgboost")


class RUXXGBClassifier(_RUGBASE):
    """
    A classifier that extracts and optimizes decision rules from a trained
    XGBoost ensemble model to create a compact and interpretable model.
    This process involves translating the ensemble's trees into a set of rules and
    using optimization to balance model accuracy and interpretability. The complexity
    of the resulting rule-based model is  controlled through a penalty parameter.
    """

    def __new__(cls, *args, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for this class but is not installed.",
                "Please install it with 'pip install xgboost'",
            )
        instance = super(RUXXGBClassifier, cls).__new__(cls)
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
        trained_ensemble : xgboost.XGBClassifier
            The trained XGBoost ensemble model from which rules will be extracted.
            The model should already be trained on the dataset.

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
            The minimum weight threshold for including a rule in the final model

        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducible results.
        """
        ### LAZY IMPORT
        from xgboost import Booster

        if not isinstance(trained_ensemble, Booster):
            if hasattr(trained_ensemble, "get_booster"):
                if not isinstance(trained_ensemble.get_booster(), Booster):
                    raise TypeError(
                        "trained_ensemble is not an instance of XGBClassifier."
                    )
                else:
                    self.trained_ensemble = trained_ensemble
            else:
                raise TypeError("trained_ensemble is not an instance of XGBoost")
        else:
            raise TypeError(
                "XGBoost Booster instance not supported yet. Use XGBClassifier."
            )

        super().__init__(
            threshold=threshold,
            random_state=random_state,
            rule_cost=rule_cost,
            solver=solver,
            class_weight=class_weight,
        )

    def _get_rule(self, fit_tree: pd.DataFrame, leaf_index: int) -> Rule:
        """
        Extracts a decision rule leading to a specified leaf node from an XGBoost tree.

        Parameters
        ----------
        fit_tree : pd.DataFrame
            The decision trees represented as a DataFrame extracted from the trained XGBoost model.
        leaf_index : int
            The ID of the leaf node for which to construct the decision rule.

        Returns
        -------
        Rule
            An object representing the decision rule leading to the specified leaf node,
            composed of clauses that define the decision path.
        """
        # Initialize the rule
        return_rule = Rule()

        if fit_tree.shape[0] <= 1:
            return return_rule

        while True:
            # Find the parent node of the current leaf
            parent = fit_tree.loc[
                np.any(fit_tree.loc[:, ["Yes", "No"]] == leaf_index, axis=1), "Node"
            ].values[0]

            # Extract information about the decision at the parent node
            feature = int(fit_tree.loc[fit_tree.Node == parent, "Feature"].values[0])

            threshold = fit_tree.loc[fit_tree.Node == parent, "Split"].values[0]
            is_left = (
                fit_tree.loc[fit_tree.Node == parent, "Yes"].values[0] == leaf_index
            )
            missing = (
                fit_tree.loc[fit_tree.Node == parent, "Missing"].values[0] == leaf_index
            )

            ub = threshold if is_left else np.inf
            lb = -np.inf if is_left else threshold
            na = missing

            return_rule.add_clause(feature, ub, lb, na)

            # If we reached the root of the tree, break the loop
            if parent == 0:
                break

            # Move up the tree
            leaf_index = parent

        return return_rule

    def _get_matrix(
        self,
        y: np.ndarray,
        vec_y: np.ndarray,
        fit_tree: pd.DataFrame,
        treeno: int,
        y_rules: np.ndarray,
    ):
        """
        Populates the matrices for the optimization problem based on a single XGBoost tree.

        Parameters
        ----------
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector, suitable for the optimization problem.
        fit_tree : pd.DataFrame
            A single decision tree's structure from XGBoost, represented as a DataFrame.
        treeno : int
            The index of the current tree within the ensemble.
        y_rules : np.ndarray
            The array of leaf indices for each sample in the training data, determined
            by the current tree.
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

            # Calculate the distribution of the samples in the leaf across the classes
            sdist = counts
            self.rule_info_[col] = (treeno, leafno, label, sdist)
            col += 1

    def _get_matrices(self, x: np.ndarray, y: np.ndarray, vec_y: np.ndarray):
        """
        Generates the coefficient matrices for the optimization problem from
        all trees in the XGBoost ensemble.

        Parameters
        ----------
        x : np.ndarray
            The feature matrix of the training data.
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector.
        """
        y_rules = self.trained_ensemble.apply(x).astype(np.intp)
        for treeno, fit_tree in enumerate(self.decision_trees_.values()):
            self._get_matrix(y, vec_y, fit_tree, treeno, y_rules)

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        """
        Fits the RUXXGBClassifier to the training data, optimizing
        the extracted rules for a balance between accuracy and interpretability.

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
        RUXXGBClassifier
            The fitted model, ready for making predictions.
        """
        x, y = check_inputs(x, y)

        sample_weight = self._get_sample_weight(sample_weight, y)

        # If the model has been fitted before, clean it up
        if self._is_fitted:
            self._cleanup()

        # Fills the fitted decision trees.
        out = self.trained_ensemble.get_booster().trees_to_dataframe()
        pattern = re.compile(r"-(\d+)")
        columns = ["Yes", "No", "Missing"]
        out[columns] = out[columns].map(
            lambda x: (
                int(pattern.search(x).group(1))
                if pd.notna(x) and pattern.search(x)
                else None
            )
        )
        out.Feature = out.Feature.str.lstrip("f")
        self.decision_trees_ = {
            treeno: out.loc[out.Tree == treeno] for treeno in out.Tree.unique()
        }

        self._get_class_infos(y)
        vec_y = self._preprocess(y)

        self._get_matrices(x=x, y=y, vec_y=vec_y)

        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        ws, *_ = self.solver(
            coefficients=self.coefficients_,
            k=self.k_,
            sample_weight=sample_weight,
            normalization_constant=normalization_constant,
            rng=self._rng
        )

        self._fill_rules(ws)
        self._is_fitted = True

        return self
