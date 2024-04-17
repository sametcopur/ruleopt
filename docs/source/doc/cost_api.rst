Rule Cost
==============

These modules provide a set of classes for calculating costs associated with different sets of rules. These cost calculations are integral to optimization problems, where the cost of a each specific rule is multiplied by the rule's weight and added to the total error rate for minimization.

.. autoclass:: ruleopt.rule_cost.Length
   :undoc-members:

.. autoclass:: ruleopt.rule_cost.Gini
   :undoc-members:

.. autoclass:: ruleopt.rule_cost.Mixed
   :members:  __init__,
   :undoc-members:

.. autoclass:: ruleopt.rule_cost.MixedSigmoid
   :members:  __init__,
   :undoc-members:

