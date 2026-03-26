# Understanding Model Decisions with Explainer

The `Explainer` class in the `ruleopt` library provides a tool for interpreting the decisions made by the trained models like `RUGClassifier`. The `Explainer` facilitates this by breaking down the predictions into understandable rules and metrics. This guide focuses on demonstrating how to use the `Explainer` alongside `RUGClassifier` to gain insights into the classification decisions on the Iris dataset.

## Step 1: Import Libraries

First, import the necessary libraries:

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini
from ruleopt.explainer import Explainer
from ruleopt.solver import HiGHSSolver
```

## Step 2: Prepare the Data

Load the Iris dataset and split it into training and testing sets:

```python
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
```

## Step 3: Initialize and Train the Classifier

Set up the `RUGClassifier` with specific parameters and fit it to the training data:

```python
# Define tree parameters, solver and rule_cost
tree_parameters = {"max_depth": 3, "class_weight": "balanced"}
solver = HiGHSSolver()
rule_cost = Gini()

# Initialize the RUGClassifier with specific parameters
rug = RUGClassifier(
    solver=solver,
    max_rmp_calls=20,
    rule_cost=rule_cost,
    **tree_parameters,
)

rug.fit(X_train, y_train)
```

## Step 4: Utilize the Explainer

Initialize the `Explainer` with the fitted `RUGClassifier`:

```python
exp = Explainer(rug)
```

In all methods, the `info` parameter can be set to `True` to collect additional output information. For more details, you can check the API reference.

### Retrieve Rule Details

Retrieve and print details for all rules:

```python
rule_details = exp.retrieve_rule_details(feature_names=["sepal length", "sepal width", "petal length", "petal width"], info=True)
```

```
+-------------------------------------------------------------+
| Rule 0  |  Class: 2  |  Weight: 1.0000                      |
+-------------------------------------------------------------+
| single | -inf < sepal length <= 6.35             | not null |
| single | 1.55 < petal width <= inf               | or null  |
| single | -inf < sepal width <= 3.10              | or null  |
+-------------------------------------------------------------+
+-------------------------------------------------------------+
| Rule 1  |  Class: 2  |  Weight: 0.6667                      |
+-------------------------------------------------------------+
| single | -inf < petal width <= 1.55              | not null |
| single | 4.90 < petal length <= inf              | or null  |
| single | -inf < sepal width <= 3.10              | or null  |
+-------------------------------------------------------------+
```

Rules are displayed in a table format. Each row shows the clause type (`single` for single-feature, `oblique` for multi-feature), the condition, and null-handling behavior. When `use_oblique=True` is used, oblique clauses appear as linear combinations:

```
+----------------------------------------------------------------------+
| Rule 0  |  Class: 1  |  Weight: 1.0000                               |
+----------------------------------------------------------------------+
| single  | -inf < petal length <= 5.05                      | or null |
| oblique | -0.53*sepal width + 1.00*petal width < 0.19                |
| oblique | 0.13*sepal length + -1.00*petal width < -0.08              |
+----------------------------------------------------------------------+
```

The returned dictionary also includes `n_clauses` and `n_oblique_clauses` fields for each rule.

### Find Applicable Rules for Test Samples

Identify rules applicable to samples in the test set:

```python
applicable_rules = exp.find_applicable_rules_for_samples(X_test, feature_names=["sepal length", "sepal width", "petal length", "petal width"], info=True)
```

By applying the classifier's rules to the test samples, we identify which rules are activated for individual instances and observe their respective weights.

```
Rules for instance 0
+--------------------------------------------------+
| Rule 2  |  Class: 1  |  Weight: 0.6667           |
+--------------------------------------------------+
| single | -inf < petal width <= 1.75   | or null  |
| single | 5.40 < sepal length <= 6.95  | or null  |
+--------------------------------------------------+
+--------------------------------------------------+
| Rule 5  |  Class: 1  |  Weight: 0.4444           |
+--------------------------------------------------+
| single | -inf < petal length <= 4.95  | or null  |
| single | 0.80 < petal width <= 1.75   | or null  |
+--------------------------------------------------+

Rules for instance 1
+--------------------------------------------------+
| Rule 2  |  Class: 1  |  Weight: 0.6667           |
+--------------------------------------------------+
| single | -inf < petal width <= 1.75   | or null  |
| single | 5.40 < sepal length <= 6.95  | or null  |
+--------------------------------------------------+
+---------------------------------------------------+
| Rule 4  |  Class: 0  |  Weight: 0.4444            |
+---------------------------------------------------+
| single | -inf < petal width <= 0.80   | not null  |
+---------------------------------------------------+
```

### Summarize Rule Metrics

Get a summary of rule complexity and count:

```python
rule_metrics_summary = exp.summarize_rule_metrics(info=True)
```

This summary provides an overview by detailing the total number of rules generated, the average rule length, and other metrics. 

```
Total number of rules: 7
Average rule length: 2.00
Total clauses: 12 single-feature, 8 oblique
```

The clause breakdown (single-feature vs oblique) is shown when the model contains oblique rules.

### Evaluate Rule Coverage Metrics

Understand how well the rules cover the test dataset:

```python
rule_coverage_metrics = exp.evaluate_rule_coverage_metrics(X_test, info=True)
```

This evaluation assesses the effectiveness and coverage of the rules across the test dataset, highlighting the number of instances not covered by any rule (they are predicted as belonging to the majority class), the average number of rules applicable per sample, and the average length of these rules.

```markdown
Number of instances not covered by any rule: 0
Average number of rules per sample: 1.75
Average length of rules per sample: 1.75
```

## Conclusion

This example shows how to classify the Iris dataset using `RUGClassifier` and analyzes the resulting rules using the `Explainer`. For more information on `ruleopt` and its features, visit the official documentation.
