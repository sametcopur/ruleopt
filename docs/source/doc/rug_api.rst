RUGClassifier
==============

``RUGClassifier`` is a machine learning model designed for classification tasks, emphasizing the generation and optimization of rules.

The unique aspect of the ``RUGClassifier`` lies in its iterative process of refining the rule set. It starts with a basic decision tree fitted on the original dataset. As the process unfolds, more trees are fitted on subsets of the data, which are weighted according to the solutions of linear programming problems. This method concentrates the learning on areas where the model currently underperforms, ensuring that subsequent iterations focus on improving these weak spots.

This classifier operates by solving Restricted Master Problems (RMPs) to iteratively enhance its objective function, which aims to find a balance between model accuracy and the simplicity of the rules it generates. This balance is crucial for maintaining the interpretability of the model while striving for high performance.

The approach allows for a detailed tuning of the model through various parameters such as penalty parameters for controlling the balance between model complexity and accuracy, costs associated with rules to manage their complexity, and thresholds for determining the significance of rules in the final model.

In essence, ``RUGClassifier`` is built to offer a blend of accuracy and interpretability in classification tasks, making it suitable for applications where understanding the model's decision-making process is as important as the accuracy of its predictions.

Oblique Splits
--------------

``RUGClassifier`` supports oblique (multi-feature) splits through the ``use_oblique`` and ``n_pair`` parameters. When enabled, decision tree nodes can split on linear combinations of features (e.g., ``0.73*x1 + -1.00*x2 < 0.15``) instead of single-feature thresholds only.

Note that enabling oblique splits reduces interpretability, since each clause involves a weighted combination of multiple features rather than a simple threshold on a single feature. To minimize this trade-off, we recommend setting ``n_pair=2`` so that each oblique clause combines at most two features.

.. autoclass:: ruleopt.RUGClassifier
   :members: __init__, fit, predict, predict_proba, is_fitted, decision_trees, decision_rules, rule_info, coefficients, majority_class, majority_probability, k, classes, rule_columns
   :undoc-members:
   :member-order: bysource

