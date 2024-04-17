"""
Example: Converting XGBoost Classifier to RUXXGBClassifier on the Iris Dataset

This example demonstrates how to train a XGBoost classifier on the Iris dataset 
and subsequently convert it into a RUXXGBClassifier using ruleopt. This process 
allows for the interpretation of the XGBoost model through rule-based explanations. 
A Gini cost function is used to balance the simplicity of the rules with their 
accuracy, enhancing interpretability without significantly sacrificing performance.
"""

# Import necessary libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from ruleopt import RUXXGBClassifier
from ruleopt.rule_cost import Gini


def main():
    # Set a random state for reproducibility
    random_state = 42

    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # Initialize and fit an XGBoost classifier
    xgb_classifier = XGBClassifier(random_state=random_state)
    xgb_classifier.fit(X_train, y_train)

    # Initialize a RUXXGBClassifier with the trained XGBoost classifier
    # Use a gini rule cost with weight 0.5 to balance simplicity and accuracy
    
    rux = RUXXGBClassifier(xgb_classifier, random_state=random_state)

    # Fit the RUXXGBClassifier to the training data for rule extraction
    rux.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = rux.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
