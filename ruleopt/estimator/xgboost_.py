from __future__ import annotations
import json
import numpy as np
from numpy.typing import ArrayLike

from .base import _RUGBASE
from ..aux_classes import Rule
from ..rule_cost import Gini
from ..utils import check_module_available, check_inputs
from ..solver import HiGHSSolver


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
        solver=HiGHSSolver(),
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

        solver : OptimizationSolver, default=HiGHSSolver()
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

    @staticmethod
    def _parse_tree_json(tree_dict: dict) -> dict:
        """Flatten a JSON tree dict into a node lookup: {nodeid: node_info}."""
        nodes = {}
        parent_map = {}

        def traverse(node, parent_id=None, is_left=None):
            nid = node["nodeid"]
            info = {"nodeid": nid}

            if "leaf" in node:
                info["is_leaf"] = True
            else:
                info["is_leaf"] = False
                info["feature"] = int(node["split"].lstrip("f"))
                info["threshold"] = float(node["split_condition"])
                info["yes"] = node["yes"]
                info["no"] = node["no"]
                info["missing"] = node.get("missing", node["no"])

            nodes[nid] = info

            if parent_id is not None:
                parent_map[nid] = (parent_id, is_left)

            if "children" in node:
                for child in node["children"]:
                    child_is_left = (child["nodeid"] == node["yes"])
                    traverse(child, nid, child_is_left)

        traverse(tree_dict)
        return nodes, parent_map

    def _get_rule(self, fit_tree: tuple, leaf_index: int) -> Rule:
        """
        Extracts a decision rule leading to a specified leaf node from an XGBoost tree.

        Parameters
        ----------
        fit_tree : tuple
            A (nodes, parent_map) tuple from _parse_tree_json.
        leaf_index : int
            The ID of the leaf node for which to construct the decision rule.

        Returns
        -------
        Rule
            An object representing the decision rule leading to the specified leaf node,
            composed of clauses that define the decision path.
        """
        return_rule = Rule()
        nodes, parent_map = fit_tree

        if len(nodes) <= 1:
            return return_rule

        current = leaf_index
        while current in parent_map:
            parent_id, is_left = parent_map[current]
            parent = nodes[parent_id]

            feature = parent["feature"]
            threshold = parent["threshold"]
            missing = (parent["missing"] == current)

            ub = threshold if is_left else np.inf
            lb = -np.inf if is_left else threshold

            return_rule.add_clause(feature, ub, lb, missing)

            current = parent_id

        return return_rule

    def _get_matrix(
        self,
        y: np.ndarray,
        vec_y: np.ndarray,
        fit_tree: tuple,
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
        fit_tree : tuple
            A (nodes, parent_map) tuple from _parse_tree_json.
        treeno : int
            The index of the current tree within the ensemble.
        y_rules : np.ndarray
            The array of leaf indices for each sample in the training data, determined
            by the current tree.
        """
        if self.coefficients_.cols.shape[0] == 0:
            col = 0
        else:
            col = np.max(self.coefficients_.cols) + 1

        y_rules = y_rules[:, treeno]

        for leafno in np.unique(y_rules):
            covers = np.where(y_rules == leafno)[0]
            leaf_y_vals = y[covers]

            counts_full = np.bincount(leaf_y_vals, minlength=self.k_)
            counts = counts_full[counts_full > 0]

            label = int(np.argmax(counts_full))

            label_vector = np.full((self.k_,), -1 / (self.k_ - 1))
            label_vector[label] = 1

            fill_ahat = np.dot(vec_y[covers, :], label_vector)

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

            self.coefficients_.costs = np.concatenate(
                (self.coefficients_.costs, [cost])
            )

            self.rule_info_[col] = (treeno, leafno, label, counts_full)
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

        if self._is_fitted:
            self._cleanup()

        dump = self.trained_ensemble.get_booster().get_dump(dump_format="json")
        for treeno, tree_json in enumerate(dump):
            tree_dict = json.loads(tree_json)
            self.decision_trees_[treeno] = self._parse_tree_json(tree_dict)

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
