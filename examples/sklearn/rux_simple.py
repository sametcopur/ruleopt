# Import necessary libraries
"""
Example: Using RUXClassifier with the Iris Dataset

This example demonstrates how to use the RUXClassifier from the ruleopt 
library to classify instances of the Iris dataset. The Iris dataset is a 
classic and easy-to-use dataset for classification tasks. Here, we use the 
Gini index as our cost function for the rule generation process.

This example demonstrates how to train a RandomForestClassifier on the Iris
dataset and subsequently convert it into a RUXClassifier using ruleopt. This 
process allows for the interpretation of the RandomForestClassifier model 
through rule-based explanations. Here, we use the Gini index as our cost 
function for the rule extraction process.
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from ruleopt import RUXClassifier
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

    # Initialize a RandomForestClassifier as the base classifier
    rfc = RandomForestClassifier(random_state=random_state)
    rfc.fit(X_train, y_train)

    rule_cost = Gini()
    solver = ORToolsSolver(penalty=3)
    
    # Initialize the RUXClassifier with specific parameters
    rux = RUXClassifier(
        rfc,  # Specify the base estimator
        random_state=random_state,
        solver = solver,
        rule_cost=rule_cost,
    )

    # Fit the RUXClassifier to the training data
    rux.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = rux.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
