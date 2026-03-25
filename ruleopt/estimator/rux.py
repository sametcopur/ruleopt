from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from numpy.typing import ArrayLike

from .base import _RUGBASE
from ..aux_classes import Rule
from ..rule_cost import Gini
from ..utils import check_inputs
from ..solver import HiGHSSolver


class RUXClassifier(_RUGBASE):
    """
    RUXClassifier aims to build a compact and interpretable model
    by employing rule-based learning extracted from a trained scikit-learn
    ensemble model such as Random Forests, Gradient Boosting Machines,
    and Extra-Trees Classifiers.
    """

    def __init__(
        self,
        trained_ensemble,
        *,
        solver=HiGHSSolver(),
        rule_cost=Gini(),
        class_weight: dict | str | None = None,
        threshold: float = 1.0e-6,
        random_state: int | None = None,
    ):
        if not isinstance(trained_ensemble, (GradientBoostingClassifier, ForestClassifier)):
            raise TypeError(
                "trained_ensemble must be an instance of "
                "sklearn.ensemble.GradientBoostingClassifier, "
                "sklearn.ensemble.RandomForestClassifier, "
                "or sklearn.ensemble.ExtraTreesClassifier."
            )

        self.trained_ensemble = trained_ensemble
        super().__init__(
            threshold=threshold,
            random_state=random_state,
            rule_cost=rule_cost,
            solver=solver,
            class_weight=class_weight,
        )

    # ── Rule extraction (sklearn tree_ based) ─────────────────────

    @staticmethod
    def _build_node_info(fit_tree: DecisionTreeClassifier) -> dict | None:
        if fit_tree.tree_.feature[0] == -2:
            return None

        tree = fit_tree.tree_
        left = tree.children_left
        right = tree.children_right
        missing_left = tree.missing_go_to_left

        node_info = {
            node_id: (parent, True, bool(missing_left[parent]))
            for parent, node_id in enumerate(left)
        }
        node_info.update({
            node_id: (parent, False, not bool(missing_left[parent]))
            for parent, node_id in enumerate(right)
        })
        return node_info

    def _get_rule(self, fit_tree: DecisionTreeClassifier, nodeid: int, node_info: dict = None) -> Rule:
        return_rule = Rule()

        if node_info is None:
            node_info = self._build_node_info(fit_tree)
            if node_info is None:
                return Rule()

        tree = fit_tree.tree_
        threshold = tree.threshold

        while nodeid != 0:
            parent, is_left, missing = node_info[nodeid]
            feature = tree.feature[parent]
            ub = threshold[parent] if is_left else np.inf
            lb = -np.inf if is_left else threshold[parent]
            return_rule.add_clause(feature, ub, lb, missing)
            nodeid = parent

        return return_rule

    # ── Matrix building ───────────────────────────────────────────

    def _get_matrix(self, x, y, vec_y, fit_tree, treeno, betas=None, normalization_constant=None):
        if self.coefficients_.cols.shape[0] == 0:
            col = 0
        else:
            col = np.max(self.coefficients_.cols) + 1

        y_rules = fit_tree.apply(x)
        preds = fit_tree.predict(x).astype(np.intp)

        no_improvement = True
        node_info = self._build_node_info(fit_tree)

        neg_val = -1 / (self.k_ - 1)
        label_vectors = np.full((self.k_, self.k_), neg_val)
        np.fill_diagonal(label_vectors, 1.0)

        rows_list, cols_list, yvals_list, costs_list = [], [], [], []

        check_reduced_cost = (betas is not None) & (normalization_constant is not None)
        k_ratio = (self.k_ - 1.0) / self.k_

        for leafno in np.unique(y_rules):
            temp_rule = self._get_rule(fit_tree, leafno, node_info)

            covers = np.where(y_rules == leafno)[0]
            counts = np.bincount(y[covers], minlength=self.k_)
            counts = counts[counts > 0]
            label = preds[covers[0]]

            fill_ahat = np.dot(vec_y[covers, :], label_vectors[label])

            cost = self._get_rule_cost(temp_rule=temp_rule, covers=covers, counts=counts, y=y)

            if check_reduced_cost:
                red_cost = np.dot(k_ratio * fill_ahat, betas[covers]) - (
                    cost * self.solver.penalty * normalization_constant
                )
            else:
                red_cost = float("inf")

            if red_cost > 0:
                rows_list.append(covers)
                cols_list.append(np.full(covers.shape[0], col, dtype=np.int32))
                yvals_list.append(np.full(covers.shape[0], fill_ahat, dtype=np.float32))
                costs_list.append(cost)
                self.rule_info_[col] = (treeno, leafno, label, counts)
                col += 1
                no_improvement = False

        if rows_list:
            self.coefficients_.rows = np.concatenate([self.coefficients_.rows] + rows_list)
            self.coefficients_.cols = np.concatenate([self.coefficients_.cols] + cols_list)
            self.coefficients_.yvals = np.concatenate([self.coefficients_.yvals] + yvals_list)
            self.coefficients_.costs = np.concatenate(
                [self.coefficients_.costs, np.array(costs_list, dtype=np.float64)]
            )

        return no_improvement

    def _get_matrices(self, x, y, vec_y):
        for treeno, fit_tree in enumerate(self.decision_trees_.values()):
            self._get_matrix(x, y, vec_y, fit_tree, treeno)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        x, y = check_inputs(x, y)
        sample_weight = self._get_sample_weight(sample_weight, y)

        if self._is_fitted:
            self._cleanup()

        tree_infos = self.trained_ensemble.estimators_
        for treeno, fit_tree in enumerate(tree_infos):
            self.decision_trees_[treeno] = (
                fit_tree[0] if isinstance(fit_tree, np.ndarray) else fit_tree
            )

        self._get_class_infos(y)
        vec_y = self._preprocess(y)
        self._get_matrices(x, y, vec_y)

        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        ws, *_ = self.solver(
            coefficients=self.coefficients_,
            k=self.k_,
            sample_weight=sample_weight,
            normalization_constant=normalization_constant,
            rng=self._rng,
        )

        self._fill_rules(ws)
        self._is_fitted = True
        return self
