# Getting Started

Welcome to `ruleopt`, a library designed to use mathematical optimization for classification problems. It is a rule generation and extraction algorithm resulting in a rule-based approach to interpretability in machine learning.

A noteworthy feature of `ruleopt` is its capability to either generate new rules or extract rules from existing ensemble models. This enables users not only to create interpretable models from scratch but also to derive rules from complex ensemble models, such as random forests and gradient boosting machines.

`ruleopt` can seamlessly work with an array of linear programming solvers, including both free and proprietary ones.

## Installation

To get started with `ruleopt`, you'll first need to install the package. You can do this easily via pip:

```bash
pip install ruleopt` 
```
Ensure you have Python 3.9 or later installed on your machine.

## Quick Start Guide

Let's dive in and see `ruleopt` in action. Here's a simple example to get you started with creating a classification model:

1.  **Import Your Library**: Begin by importing `ruleopt` into your Python script.

```python
from ruleopt import RUGClassifier
from ruleopt.cost import Gini
from ruleopt.solver import ORToolsSolver
```
1.  **Load Your Data**: Load your dataset. `ruleopt` works with data in `NumPy` arrays, `Pandas` DataFrames, and other common formats.

```python
X_train, X_test, y_train, y_test = load_data(...)
```
3.  **Create and Train Your Model**: Instantiate a classifier and use your data to train the model.

```python
# Define scikit-learn tree parameters in a dict, solver and rule_cost
tree_parameters = {"max_depth": 3, "class_weight": "balanced"}
solver = ORToolsSolver()
rule_cost = Gini()
random_state = 42

# Initialize the RUGClassifier with specific parameters
classifier = RUGClassifier(
    solver=solver,
    rule_cost=rule_cost,
    random_state=random_state,
    **tree_parameters,
)

classifier.fit(X_train, y_train)
```
4.  **Make Predictions**: Once the model is trained, you can use it to make predictions on new data.

```python
predictions = classifier.predict(X_test) 
```
## Next Steps

Congratulations on running your first model with `ruleopt`! To dive deeper into the capabilities of our library, explore the following resources:

-   **Detailed Tutorials**: Walk through more examples and learn different techniques in [our tutorials section on Github](https://github.com/sametcopur/ruleopt/tree/main/examples).
-   **API Reference**: Get detailed information on every function, class, and method provided by `ruleopt` in the API Reference.

## Important Features

### Handling Missing Values

`ruleopt` is able to work natively with datasets that contain missing values. Thus, the preliminary steps to address or remove these missing values are not needed.

### Integration with Popular Machine Learning Frameworks

`ruleopt` works not only as a standalone solution, but it also integrates with leading machine learning frameworks:

- **Integration with scikit-learn Ensemble Models**: `ruleopt` enables the straightforward extraction of rules from `scikit-learn` ensemble models, such as Random Forests, Gradient Boosting Machines, and Extra-Trees Classifiers. This capability merges the interpretability objective of `ruleopt` with the predictive power and robustness of ensemble techniques.

```python
from ruleopt import RUXCLassifier
from sklearn.ensemble import RandomForestClassifier

# Train a RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Extract rules from the trained model
rux_classifier = RUXCLassifier(rf_classifier, solver = solver)
rux.fit(X_train, y_train)
```
-   **Integration with XGBoost and LightGBM**: `ruleopt` can also extract rules from models trained with `XGBoost` and  `LightGBM`. Using these rules, `ruleopt` trains a model with optimization to assign weights to rules and improve interpretability.

```python
from ruleopt import RUXLGBMClassifier #RUXXGBClassifier
import lightgbm as lgb

# Train an XGBoost model
lgb_model = lgb.LGBMlassifier()
lgb_model.fit(X_train, y_train)

# Extract rules from the trained LightGBM model
rux_lgbm_classifier = RUXLGBClassifier(lgb_model, solver = solver)
rux_lgbm_classifier.fit(X_train, y_train)
```
