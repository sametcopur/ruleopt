"""
Example: Using RUGClassifier with the Iris Dataset

This example demonstrates how to use the RUGClassifier from the ruleopt 
library to classify instances of the Iris dataset. The Iris dataset is a 
classic and easy-to-use dataset for classification tasks. Here, we use the 
Gini index as our cost function for the rule generation process.
"""

# Import necessary libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini
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

    solver = ORToolsSolver()
    rule_cost = Gini()

    # Initialize the RUGClassifier with specific parameters
    rug = RUGClassifier(
        solver=solver,
        random_state=random_state,
        max_depth=3,
        max_rmp_calls=20,
        rule_cost=rule_cost,
    )

    # Fit the RUGClassifier to the training data
    rug.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = rug.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
