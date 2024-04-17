"""
Example: Integrating LightGBM with RUXLGBMClassifier on the Iris Dataset

This example demonstrates the process of training a LightGBM model on 
the Iris dataset and then using the RUXLGBMClassifier from ruleopt to 
interpret the LightGBM model with rule-based explanations. The Iris dataset
is used for this classification task, and the Length cost function is used 
to prioritize simpler rules.
"""

# Import necessary libraries
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from ruleopt import RUXLGBMClassifier
from ruleopt.rule_cost import Mixed
from ruleopt.solver import ORToolsSolver


def main():
    # Set a random state for reproducibility
    random_state = 42

    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create LightGBM datasets for training and testing
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Define the parameters for the LightGBM model
    params = {
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": random_state,
    }

    # Train the LightGBM model
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
    )  # Set to True to see training logs

    # Initialize a RUXLGBMClassifier with the trained LightGBM Booster
    rule_cost = Mixed(w=0.5)
    solver = ORToolsSolver()

    rux = RUXLGBMClassifier(
        gbm, solver=solver, random_state=random_state, rule_cost=rule_cost
    )

    # Fit the RUXLGBMClassifier to the training data for rule extraction
    rux.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = rux.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
