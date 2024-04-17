"""
Example: Using RUGClassifier with the Iris Dataset

This example demonstrates how to use the RUGClassifier from the ruleopt 
library to classify instances of the Iris dataset. The Iris dataset is a 
classic and easy-to-use dataset for classification tasks. Here, we use the 
Gini index as our cost function for the rule generation process.
"""

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from ruleopt import RUGClassifier, Explainer
from ruleopt.rule_cost import Length


def main():
    # Set a random state for reproducibility
    random_state = 42

    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=random_state
    )

    # Define tree parameters
    tree_parameters = {"max_depth": 3}
    rule_cost = Length()

    # Initialize the RUGClassifier with specific parameters
    rug = RUGClassifier(
        random_state=random_state,
        max_rmp_calls=3,
        rule_cost=rule_cost,
        **tree_parameters
    )

    # Fit the RUGClassifier to the training data
    rug.fit(X_train, y_train)

    # Initialize the Explainer with the fitted RUGClassifier
    exp = Explainer(rug)

    # Example usage of the Explainer's methods:

    # 1. Retrieve rule details. You can either specify indices of specific rules or leave it as None to get all rules.
    rule_details = exp.retrieve_rule_details(
        feature_names=["sepal length", "sepal width", "petal length", "petal width"],
        info=True,
    )

    # 2. Find applicable rules for samples in the test set
    applicable_rules = exp.find_applicable_rules_for_samples(
        X_test,
        threshold=0,
        feature_names=["sepal length", "sepal width", "petal length", "petal width"],
        info=True,
    )

    # 3. Summarize rule metrics to understand the overall rule complexity and count
    rule_metrics_summary = exp.summarize_rule_metrics(info=True)

    # 4. Evaluate rule coverage metrics to understand how well the rules cover the test dataset
    rule_coverage_metrics = exp.evaluate_rule_coverage_metrics(X_test, info=True)


if __name__ == "__main__":
    main()
