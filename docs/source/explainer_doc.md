# Understanding Model Decisions with Explainer

The `Explainer` class in the `ruleopt` library provides a tool for interpreting the decisions made by the trained models like `RUGClassifier`. The `Explainer` facilitates this by breaking down the predictions into understandable rules and metrics. This guide focuses on demonstrating how to use the `Explainer` alongside `RUGClassifier` to gain insights into the classification decisions on the Iris dataset.

## Step 1: Import Libraries

First, import the necessary libraries:

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from ruleopt import RUGClassifier
from ruleopt.cost import Gini
from ruleopt.explainer import Explainer
from ruleopt.solver import ORToolsSolver
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
solver = ORToolsSolver()
rule_cost = Gini()

# Initialize the RUGClassifier with specific parameters
rug = RUGClassifier(
    solver=solver,
    tree_parameters=tree_parameters,
    max_rmp_calls=20,
    rule_cost=rule_cost,
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

```markdown
RULE 0:
-inf      < sepal length <= 6.35      and not null
1.55      < petal width <= inf       or null
-inf      < sepal width <= 3.10      or null
Class: 2
Scaled rule weight: 1.0000

RULE 1:
-inf      < petal width <= 1.55      and not null
4.90      < petal length <= inf       or null
-inf      < sepal width <= 3.10      or null
Class: 2
Scaled rule weight: 0.6667
```

This step reveals the specific conditions or thresholds for various features like _sepal length_, _sepal width_, _petal length_, and _petal width_, that define each rule, along with the class association and the scaled rule weight. For example, Rule 0 defines conditions across multiple features and assigns these to Class 2 with a full rule weight, indicating a strong influence in classification decisions for this rule.

### Find Applicable Rules for Test Samples

Identify rules applicable to samples in the test set:

```python
applicable_rules = exp.find_applicable_rules_for_samples(X_test, feature_names=["sepal length", "sepal width", "petal length", "petal, width"], info=True)
```

By applying the classifier rules to the test samples, we identify which rules are activated for individual instances and observe their respective weights.

```markdown
Rules for instance 0
RULE 2:
-inf      < petal width <= 1.75      or null
5.40      < sepal length <= 6.95      or null
Class: 1
Scaled rule weight: 0.6667

RULE 5:
-inf      < petal length <= 4.95      or null
0.80      < petal width <= 1.75      or null
Class: 1
Scaled rule weight: 0.4444

Rules for instance 1
RULE 2:
-inf      < petal width <= 1.75      or null
5.40      < sepal length <= 6.95      or null
Class: 1
Scaled rule weight: 0.6667

RULE 4:
-inf      < petal width <= 0.80      and not null
Class: 0
Scaled rule weight: 0.4444
```

### Summarize Rule Metrics

Get a summary of rule complexity and count:

```python
rule_metrics_summary = exp.summarize_rule_metrics(info=True)
```

This summary provides an overview by detailing the total number of rules generated, the average rule length, and other metrics. 

```markdown
Total number of rules: 7
Average rule length: 2.00
```

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

This example shows how to classify the Iris dataset using `RUGClassifier` and analyze the resulting rules using the `Explainer`. For more information on `ruleopt` and its features, visit the official documentation.