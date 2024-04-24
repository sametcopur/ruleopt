from sklearn.tree import DecisionTreeClassifier
import numpy as np
from ..base import _RUGBASE
from ...aux_classes import Rule


class _RUGSKLEARN(_RUGBASE):
    """
    The base class specialized for use with scikit-learn.

    This subclass implements the rule extraction process specifically for decision trees
    trained using scikit-learn.
    """

    def __init__(
        self,
        solver,
        rule_cost,
        class_weight,
        threshold,
        random_state,
    ):
        """
        Parameters
        ----------
        solver : OptimizationSolver
            An instance of a derived class inherits from the 'Optimization Solver' base class.
            The solver is responsible for optimizing the rule set based on the cost function
            and constraints.

        rule_cost : RuleCost or int
            Defines the cost of rules, either as a specific calculation method (RuleCost instance)
            or a fixed cost

        class_weight: dict, "balanced" or None
            A dictionary mapping class labels to their respective weights, the string "balanced"
            to automatically adjust weights inversely proportional to class frequencies,
            or None for no weights. Used to adjust the model in favor of certain classes.

        threshold : floa
            The minimum weight threshold for including a rule in the final model

        random_state : int or None
            Seed for the random number generator to ensure reproducible results.
            Defaults to None.
        """
        super().__init__(
            threshold=threshold,
            random_state=random_state,
            rule_cost=rule_cost,
            solver=solver,
            class_weight=class_weight,
        )

    def _get_rule(self, fit_tree: DecisionTreeClassifier, nodeid: int) -> Rule:
        """
        Constructs a rule from a given node in a decision tree.

        Parameters
        ----------
        fit_tree : DecisionTreeClassifier
            The fitted decision tree from which to extract the rule.
        nodeid : int
            The ID of the node from which the rule is to be extracted.

        Returns
        -------
        Rule
            An object representing the extracted rule.
        """

        # Initializing the rule to be returned
        return_rule = Rule()

        # If the first feature of the tree is -2, the rule is empty
        if fit_tree.tree_.feature[0] == -2:
            return Rule()

        # Extracting information from the tree
        tree = fit_tree.tree_
        left = tree.children_left
        right = tree.children_right
        threshold = tree.threshold
        missing_left = tree.missing_go_to_left

        # Building dictionaries to hold node information
        node_info = {
            node_id: (parent, True, bool(missing_left[parent]))
            for parent, node_id in enumerate(left)
        }
        node_info.update(
            {
                node_id: (parent, False, not bool(missing_left[parent]))
                for parent, node_id in enumerate(right)
            }
        )

        # Traversing up the tree to build the rule
        while nodeid != 0:
            parent, is_left, missing = node_info[nodeid]

            feature = tree.feature[parent]
            ub = threshold[parent] if is_left else np.inf
            lb = -np.inf if is_left else threshold[parent]
            na = missing

            return_rule.add_clause(feature, ub, lb, na)
            nodeid = parent

        return return_rule

    def _get_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vec_y: np.ndarray,
        fit_tree: DecisionTreeClassifier,
        treeno: int,
        betas: np.ndarray = None,
        normalization_constant: np.ndarray = None
    ) -> None:
        """
        Generates matrices for optimization.

        Parameters
        ----------
        x : np.ndarray
            The feature matrix of the training data.
        y : np.ndarray
            The target vector of the training data.
        vec_y : np.ndarray
            The preprocessed target vector, adjusted for optimization.
        fit_tree : DecisionTreeClassifier
            A fitted decision tree model from which to extract rules.
        treeno : int
            An identifier for the decision tree within an ensemble.
        """
        # If the coefficients matrix is empty, start from the first column
        if self.coefficients_.cols.shape[0] == 0:
            col = 0
        else:
            # Otherwise, start from the next available column
            col = np.max(self.coefficients_.cols) + 1

        # Get the leaf node for each sample in x
        y_rules = fit_tree.apply(x)
        preds = fit_tree.predict(x).astype(np.intp)

        no_improvement = True

        # Iterate over unique leaf nodes
        for leafno in np.unique(y_rules):
            temp_rule = self._get_rule(fit_tree, leafno)

            if hasattr(self, "_temp_rules"):
                if temp_rule in self._temp_rules:
                    continue

            covers = np.where(y_rules == leafno)[0]
            leaf_y_vals = y[covers]  # y values of the samples in the leaf

            _, counts = np.unique(leaf_y_vals, return_counts=True)

            # Identify the majority class in the leaf
            label = preds[covers][0]

            # Create a vector for this label
            label_vector = np.full((self.k_,), -1 / (self.k_ - 1))
            label_vector[label] = 1

            # Calculate fill_ahat, which will be used to update yvals in the coefficients matrix
            fill_ahat = np.dot(vec_y[covers, :], label_vector)
            
            cost = self._get_rule_cost(
                    temp_rule=temp_rule,
                    covers=covers,
                    counts=counts,
                    y=y,
                )

            if (betas is not None) & (normalization_constant is not None):
                red_cost = np.dot(
                    np.multiply(((self.k_ - 1.0) / self.k_), fill_ahat),
                    betas[
                        covers
                    ],  # (betas if sample_weight is None else betas * sample_weight),
                ) - (cost * self.solver.penalty * normalization_constant)

            else:
                red_cost = float("inf")

            if red_cost > 0:
                covers_fill = np.full((covers.shape[0],), fill_ahat, dtype=np.float32)
                covers_col = np.full((covers.shape[0],), col, dtype=np.int32)


                self.coefficients_.rows = np.concatenate(
                    (self.coefficients_.rows, covers)
                )
                self.coefficients_.cols = np.concatenate(
                    (self.coefficients_.cols, covers_col)
                )
                self.coefficients_.yvals = np.concatenate(
                    (self.coefficients_.yvals, covers_fill)
                )

                # Append the cost to the costs in the coefficients matrix
                self.coefficients_.costs = np.concatenate(
                    (self.coefficients_.costs, [cost])
                )

                # Calculate the distribution of the samples in the leaf across the classes
                sdist = counts
                self.rule_info_[col] = (treeno, leafno, label, sdist)
                col += 1
                
                no_improvement = False
                
        return no_improvement
