Explainer
==============

``Explainer`` class provides a set of methods for interpreting and understanding the behavior of fitted estimators such as ``RUGClassifier``, ``RUXClassifier``, ``RUXLGBMClassifier``, or ``RUXXGBClassifier``.

Rules are displayed in a table format with aligned columns. Each clause is labelled as ``single`` (single-feature threshold) or ``oblique`` (multi-feature linear combination). When oblique rules are present, ``summarize_rule_metrics`` also reports a breakdown of clause types.

.. autoclass:: ruleopt.Explainer
   :members: __init__, retrieve_rule_details, find_applicable_rules_for_samples, summarize_rule_metrics, evaluate_rule_coverage_metrics
   :undoc-members:

