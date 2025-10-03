from __future__ import annotations
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from ..estimator.base import _RUGBASE

from numpy.typing import ArrayLike


class Explainer:
    """
    Initializes the Explainer with a given estimator. The estimator must be fitted and
    of a type that inherits from _RUGBASE, such as RUGClassifier, RUXClassifier,
    RUXLGBMClassifier, or RUXXGBClassifier.
    """

    def __init__(self, estimator: _RUGBASE) -> None:
        """
        Parameters
        ----------
        estimator : ruleopt.estimator instance
            A fitted estimator of a type that inherits from _RUGBASE.
        """
        if not isinstance(estimator, _RUGBASE):
            raise TypeError(
                "Estimator should be an instance of a class inheriting from _RUGBASE, ",
                f"not {type(estimator)}",
            )

        check_is_fitted(estimator, attributes=["_is_fitted"])

        self.estimator = estimator

    def retrieve_rule_details(
        self,
        feature_names: list | None = None,
        indices: list | None = None,
        info: bool = True,
    ) -> dict:
        """
        Retrieves and optionally prints detailed information about the specified rules.
        If indices are provided, information for those specific rules is returned.
        Otherwise, information for all rules is returned.

        Parameters
        ----------
        feature_names : list or None, default=None
            List of feature names for more readable rule descriptions. If None, indices are used.

        indices :list or None, default=None
            Indices of the rules to retrieve. If None, retrieves all rules.

        info : bool, default=True
            If True, prints the rules' details in a human-readable format.

        Returns
        -------
        dict
            A dictionary mapping each rule index to its details, including label, weight,
            rule description, and statistical distribution. If a rule has no conditions,
            it sets the majority class.
        """
        rules = self.estimator.decision_rules_
        if indices is not None:
            # Validate indices
            max_index = max(rules.keys())
            indices = [i for i in indices if i <= max_index]
            if len(indices) < len(set(indices)):
                warnings.warn("Some specified indices are out of range.")
        else:
            indices = list(rules.keys())

        return_dict = {}
        for indx in indices:
            rule = rules.get(indx, None)
            if rule is None:
                continue  # Skip if rule does not exist, alternative to raising an error
            rule_details = {
                "label": int(rule.label),
                "weight": float(rule.weight),
                "rule": (
                    rule.to_dict(feature_names)
                    if rule
                    else "No Rule: Set Majority Class"
                ),
                "sdist": rule.sdist.tolist(),
            }
            return_dict[indx] = rule_details
            if info:
                self._display_rule_info(indx, rule, feature_names)

        return return_dict

    def find_applicable_rules_for_samples(
        self,
        x: ArrayLike,
        threshold: float = 0.0,
        feature_names: list | None = None,
        info: bool = True,
    ) -> list:
        """
        Identifies which rules apply to each instance in the provided input data
        based on a given threshold. Optionally, prints detailed information about
        each rule that applies to the instances.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to dtype=np.float32.

        threshold : float, default=0.0
            Minimum rule weight threshold for considering a rule as covering an instance.

        feature_names : list or None, default=None
            List of feature names for more readable rule descriptions. If None, indices are used.

        info : bool, default=True
            If True, prints the details of the applicable rules for each instance.

        Returns
        -------
        List[List[int]]
            A list of lists, where each inner list contains the indices of rules that cover
            the corresponding instance in `x`.
        """
        x = check_array(np.asarray(x, dtype=np.float32), ensure_all_finite="allow-nan")

        selected_rules = []
        rule_indexs = []
        for rule_index, rule in self.estimator.decision_rules_.items():
            if rule.weight >= threshold:
                selected_rules.append(rule)
                rule_indexs.append(rule_index)

        rule_matrix = np.zeros((x.shape[0], len(selected_rules)), dtype=np.int8)

        for rule_index, rule in enumerate(selected_rules):
            rule_matrix[:, rule_index] = rule.check_rule(x)

        index_list = [np.where(row > 0)[0].tolist() for row in rule_matrix]

        if info:
            for i, applied_rules in enumerate(index_list):
                print(f"Rules for instance {i}")

                if len(applied_rules) > 0:
                    for rule_id in applied_rules:
                        self._display_rule_info(
                            rule_indexs[rule_id], selected_rules[rule_id], feature_names
                        )

                else:
                    print(
                        f"No rules applicable. Filled with majority class.\nClass: {self.estimator.majority_class_}\n"
                    )

        return index_list

    def _display_rule_info(self, indx, rule, feature_names=None) -> None:
        """
        Prints the details of a specified rule in a human-readable format. This
        method is designed for internal use to support other public methods that
        may require detailed rule information to be printed.

        Parameters
        ----------
        indx : int
            Index of the rule being printed.

        rule : Rule object
            The rule whose details are to be printed.

        feature_names : Optional[List[str]], default=None
            List of feature names for more readable rule descriptions. If None,
            feature indices are used.
        """
        print(f"RULE {indx}:")
        rule_description = (
            rule.to_text(feature_names) if feature_names else rule.to_text()
        )
        print(rule_description)
        print(f"Class: {rule.label}")
        print(f"Scaled rule weight: {rule.weight:.4f}\n")

    def summarize_rule_metrics(self, info: bool = True) -> dict:
        """
        Calculates and optionally prints the total number of rules and the average
        rule length within the model. 
        
        If instance is not covered by any rule, rule length counts as 0. 

        Parameters
        ----------
        info : bool, default=True
            If True, prints the summary information.

        Returns
        -------
        dict
            A dictionary containing 'num_of_rules' (the total number of rules) and
            'avg_rule_length' (the average length of the rules).
        """
        num_of_rules = len(self.estimator.decision_rules_)
        avg_rule_length = np.mean(
            [len(rule) for rule in self.estimator.decision_rules_.values()]
        )

        if info:
            print(f"Total number of rules: {num_of_rules}")
            print(f"Average rule length: {avg_rule_length:.2f}")

        return {"num_of_rules": num_of_rules, "avg_rule_length": avg_rule_length}

    def evaluate_rule_coverage_metrics(self, x: ArrayLike, info: bool = True) -> dict:
        """
        Calculates metrics including the number of instances not covered by any
        rule ('num_of_missed'), the average number of rules per sample
        ('avg_num_rules_per_sample'), and the average rule length per sample
        ('avg_rule_length_per_sample'). Optionally, prints this information.
        
        If instance is not covered by any rule, rule length counts as 0. 

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to dtype=np.float32.

        info : bool, default=True
            If True, prints the calculated metrics.

        Returns
        -------
        dict
            A dictionary with calculated metrics: 'num_of_missed', 'avg_num_rules_per_sample',
            and 'avg_rule_length_per_sample'.
        """
        x = check_array(np.asarray(x, dtype=np.float32), ensure_all_finite="allow-nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            missed_values_index_, rules_per_sample_, rule_length_per_sample_ = (
                self.estimator.predict(x, predict_info=True)
            )

        results = {
            "num_of_missed": len(missed_values_index_),
            "avg_num_rules_per_sample": np.mean(rules_per_sample_),
            "avg_rule_length_per_sample": np.mean(rule_length_per_sample_),
        }

        if info:
            print(
                f"Number of instances not covered by any rule: {results['num_of_missed']}"
            )
            print(
                f"Average number of rules per sample: {results['avg_num_rules_per_sample']:.2f}"
            )
            print(
                f"Average length of rules per sample: {results['avg_rule_length_per_sample']:.2f}"
            )

        return results
