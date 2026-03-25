from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.class_weight import compute_sample_weight
from obliquetree import Classifier as ObliqueTreeClassifier
from obliquetree.utils import export_tree

from .base import _RUGBASE
from ..aux_classes import Rule
from ..rule_cost import Gini
from ..utils import check_inputs
from ..solver import HiGHSSolver


class RUGClassifier(_RUGBASE):
    """
    Rule Generation algorithm for multi-class classification. This algorithm aims at
    producing a compact and interpretable model by employing optimization-based rule learning.
    """

    def __init__(
        self,
        solver=HiGHSSolver(),
        rule_cost=Gini(),
        max_rmp_calls=10,
        threshold: float = 1.0e-6,
        random_state: int | None = None,
        class_weight: dict | str | None = None,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        ccp_alpha: float = 0.0,
        categories: list | None = None,
    ):
        """
        Parameters
        ----------
        solver : OptimizationSolver, default=HiGHSSolver()
            An instance of a derived class inherits from the 'Optimization Solver' base class.

        rule_cost : RuleCost or int, default=Gini()
            Defines the cost of rules, either as a specific calculation method (RuleCost instance)
            or a fixed cost

        max_rmp_calls : int, default=10
            Maximum number of Restricted Master Problem (RMP) iterations allowed during fitting.

        class_weight: dict, "balanced" or None, default=None
            A dictionary mapping class labels to their respective weights, the string "balanced"
            to automatically adjust weights inversely proportional to class frequencies,
            or None for no weights.

        threshold : float, default=1.0e-6
            The minimum weight threshold for including a rule in the final model.

        random_state : int or None, default=None
            Seed for the random number generator to ensure reproducible results.

        criterion : {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, nodes are expanded until
            all leaves are pure or contain fewer than min_samples_split samples.

        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.

        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning.

        categories : list or None, default=None
            List of column indices representing categorical features.
        """
        self._validate_parameters(
            max_rmp_calls, criterion, max_depth,
            min_samples_split, min_samples_leaf, ccp_alpha,
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.categories = categories

        self._temp_rules = []

    # ── Tree fitting ──────────────────────────────────────────────

    def _fit_decision_tree(self, x, y, sample_weight):
        dt = ObliqueTreeClassifier(
            random_state=int(self._rng.integers(np.iinfo(np.int16).max)),
            max_depth=self.max_depth if self.max_depth is not None else -1,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            ccp_alpha=self.ccp_alpha,
            use_oblique=False,
            categories=self.categories if self.categories is not None else [],
        )
        return dt.fit(x, y, sample_weight=sample_weight)

    # ── Rule extraction (obliquetree export_tree based) ───────────

    @staticmethod
    def _build_node_info(fit_tree) -> dict | None:
        """Flatten export_tree nested dict to node_info via pre-order traversal."""
        tree_dict = export_tree(fit_tree)["tree"]

        nodes = []

        def traverse(node):
            idx = len(nodes)
            nodes.append(node)
            if not node.get("is_leaf", False):
                node["_left_idx"] = traverse(node["left"])
                node["_right_idx"] = traverse(node["right"])
            return idx

        traverse(tree_dict)

        if len(nodes) == 1:
            return None

        node_info = {}
        features = []
        thresholds = []

        for i, node in enumerate(nodes):
            features.append(node.get("feature_idx", -1))
            thresholds.append(node.get("threshold", 0.0))
            if not node.get("is_leaf", False):
                mgl = node.get("missing_go_left", True)
                node_info[node["_left_idx"]] = (i, True, mgl)
                node_info[node["_right_idx"]] = (i, False, not mgl)

        node_info["_features"] = features
        node_info["_thresholds"] = thresholds
        return node_info

    def _get_rule(self, fit_tree, nodeid: int, node_info: dict = None) -> Rule:
        return_rule = Rule()

        if node_info is None:
            node_info = self._build_node_info(fit_tree)
            if node_info is None:
                return Rule()

        features = node_info["_features"]
        thresholds = node_info["_thresholds"]

        while nodeid != 0:
            parent, is_left, missing = node_info[nodeid]
            thresh = thresholds[parent]
            return_rule.add_clause(
                features[parent],
                thresh if is_left else np.inf,
                -np.inf if is_left else thresh,
                missing,
            )
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

            if temp_rule in self._temp_rules:
                continue

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

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None):
        x, y = check_inputs(x, y)
        sample_weight = self._get_sample_weight(sample_weight, y)

        if self._is_fitted:
            self._cleanup()

        treeno = 0
        initial_sw = compute_sample_weight(self.class_weight, y) if self.class_weight is not None else None

        fit_tree = self._fit_decision_tree(x, y, sample_weight=initial_sw)
        self.decision_trees_[treeno] = fit_tree

        self._get_class_infos(y)
        vec_y = self._preprocess(y)
        self._get_matrix(x, y, vec_y, fit_tree, treeno)

        normalization_constant = 1.0 / np.max(self.coefficients_.costs)

        ws, betas = self.solver(
            coefficients=self.coefficients_, k=self.k_,
            sample_weight=sample_weight, normalization_constant=normalization_constant,
            rng=self._rng,
        )

        for _ in range(self.max_rmp_calls):
            if np.all(betas <= 1e-6):
                break

            treeno += 1
            fit_tree = self._fit_decision_tree(x, y, sample_weight=betas)
            self.decision_trees_[treeno] = fit_tree

            if self._get_matrix(x, y, vec_y, fit_tree, treeno, betas, normalization_constant):
                break

            ws, betas = self.solver(
                coefficients=self.coefficients_, k=self.k_,
                normalization_constant=normalization_constant,
                sample_weight=sample_weight, rng=self._rng,
            )

        self._fill_rules(ws)
        self._is_fitted = True
        return self

    # ── Validation ────────────────────────────────────────────────

    @staticmethod
    def _validate_parameters(max_rmp_calls, criterion, max_depth,
                             min_samples_split, min_samples_leaf, ccp_alpha):
        if not isinstance(max_rmp_calls, (float, int)):
            raise TypeError("max_rmp_calls must be an integer.")
        if max_rmp_calls < 0:
            raise ValueError("max_rmp_calls must be a non-negative integer.")
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be one of 'gini' or 'entropy'.")
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer or None.")
        if isinstance(max_depth, int) and max_depth < 1:
            raise ValueError("max_depth must be greater than 0.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")
        if not isinstance(ccp_alpha, float) or ccp_alpha < 0.0:
            raise ValueError("ccp_alpha must be a non-negative float.")
