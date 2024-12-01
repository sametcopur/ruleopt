from __future__ import annotations

import numpy as np
from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy.typing import ArrayLike

from .base_sklearn import _RUGSKLEARN
from ...rule_cost import Gini
from ...utils import check_inputs
from ...solver import ORToolsSolver


class RUXClassifier(_RUGSKLEARN):
    """
    RUXClassifier aims to build a compact and interpretable model
    by employing rule-based learning extracted from a trained scikit-learn
    ensemble model such as Random Forests, Gradient Boosting Machines, 
    and Extra-Trees Classifiers. It allows a user to trade off between
    the complexity of the model (number of rules) and the accuracy of the model.
    """

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
        trained_ensemble : sklearn.ensemble object
            The trained scikit-learn ensemble model from which the rules will be
            extracted.

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
        if not (
            isinstance(trained_ensemble, (GradientBoostingClassifier, ForestClassifier))
        ):
            raise TypeError(
                "trained_ensemble must be an instance of ",
                "sklearn.ensemble.GradientBoostingClassifier, ",
                "sklearn.ensemble.RandomForestClassifier, ",
                "or sklearn.ensemble.ExtraTreesClassifier.",
            )

        self.trained_ensemble = trained_ensemble
        super().__init__(
            threshold=threshold,
            random_state=random_state,
            rule_cost=rule_cost,
            solver=solver,
            class_weight=class_weight,
        )

    def _get_matrices(self, x: np.ndarray, y: np.ndarray, vec_y: np.ndarray) -> None:
        """
        Prepares the optimization problem matrices based on the ensemble of decision trees.

        Parameters
        ----------
        x : np.ndarray
            The feature matrix of the training data.
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector, adjusted for optimization.
        """
        for treeno, fit_tree in enumerate(self.decision_trees_.values()):
            self._get_matrix(x, y, vec_y, fit_tree, treeno)

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        """
        Fits the RUXClassifier to the training data.

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
        RUXClassifier
            The fitted model, ready for making predictions.
        """
        x, y = check_inputs(x, y)

        sample_weight = self._get_sample_weight(sample_weight, y)

        # If the model has been fitted before, clean it up
        if self._is_fitted:
            self._cleanup()

        # Fills the fitted decision trees.
        tree_infos = self.trained_ensemble.estimators_
        for treeno, fit_tree in enumerate(tree_infos):
            self.decision_trees_[treeno] = (
                fit_tree[0] if isinstance(fit_tree, np.ndarray) else fit_tree
            )

        # Extract and set properties of the target variable
        self._get_class_infos(y)

        # Preprocess the target values
        vec_y = self._preprocess(y)

        # Calculate the coefficients and other parameters for the optimization problem
        self._get_matrices(x, y, vec_y)

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
