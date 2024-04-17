Solver
==============

The linear programming model used for optimizing rule sets can be described by the following formulation:

.. math::
   :nowrap:

   \begin{align*}
   &\text{minimize} \quad &\lambda \sum_{j \in J} c_j w_j + \sum_{i \in I} v_i \\
   &\text{subject to} \quad &\sum_{j \in J} \hat{a}_{ij} w_j + v_i \geq 1, &\quad i \in I, \\
   &&v_i \geq 0, &\quad i \in I, \\
   &&w_j \geq 0, &\quad j \in J,
   \end{align*}

where

- :math:`w_j` represents the rule weights,
- :math:`\lambda \geq 0` is a hyperparameter used to scale different units in the objective function, emphasizing the trade-off between accuracy and rule costs,
- :math:`c_j` represents the costs associated with each rule, potentially reflecting the complexity or length of the rule for promoting sparsity,
- :math:`\hat{a}_{ij}` is a measure of the classification accuracy of rule :math:`j` for sample :math:`i`, given that the sample is covered by the rule,
- :math:`v_i` is a auxiliary variable standing for :math:`v_i \geq L(\hat{y}_i(w), y_i)`, where a value of :math:`v_i = 0` indicates correct classification.

For detailed information, please refer to `our manuscript <https://arxiv.org/abs/2104.10751>`_.

.. autoclass:: ruleopt.solver.ORToolsSolver
   :members:  __init__
   :undoc-members:

.. autoclass:: ruleopt.solver.GurobiSolver
   :members:  __init__
   :undoc-members:

.. autoclass:: ruleopt.solver.CPLEXSolver
   :members:  __init__
   :undoc-members:




