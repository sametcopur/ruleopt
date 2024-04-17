"""Example: Classifying the Iris Dataset with RUGClassifier using GridSearchCV"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini, Length
from ruleopt.solver import GurobiSolver


def main():
    # Ensure reproducibility by setting a fixed random state
    random_state = 42

    # Load the Iris dataset, a popular example for classification tasks
    X, y = load_iris(return_X_y=True)

    # Initialize the Gurobi solver for optimizing the RUGClassifier
    solver = GurobiSolver()

    # Create an instance of RUGClassifier with the specified solver and random state
    rug_classifier = RUGClassifier(solver=solver, random_state=random_state)

    # Define a grid of parameters to search over
    param_grid = {
        "rule_cost": [Gini(), Length()],  # Cost functions for generating rules
        "max_rmp_calls": [3, 5, 6],  # Max number of restricted master problem calls
        "threshold": [1e-6, 1e-4, 1e-2],  # Threshold for rule inclusion
        "max_depth": [2, 3, 4],  # Max depth of the decision trees
        "min_samples_split": [
            2,
            5,
            10,
        ],  # Min number of samples required to split a node
        "min_samples_leaf": [1, 2, 4],  # Min number of samples required at a leaf node
        "min_weight_fraction_leaf": [
            0.0,
            0.1,
            0.2,
        ],  # Min weighted fraction of the sum total of weights required to be at a leaf node
        "max_features": [
            "auto",
            "sqrt",
            "log2",
            None,
        ],  # Number of features to consider when looking for the best split
        "max_leaf_nodes": [None, 10, 20, 30],  # Max number of leaf nodes
        "min_impurity_decrease": [
            0.0,
            0.1,
            0.2,
        ],  # Min impurity decrease required for a split to be considered
        "ccp_alpha": [
            0.0,
            0.01,
            0.1,
        ],  # Complexity parameter used for Minimal Cost-Complexity Pruning
    }

    # Configure GridSearchCV with the RUGClassifier, parameter grid, and other settings
    grid_search = GridSearchCV(
        estimator=rug_classifier,
        param_grid=param_grid,
        cv=3,  # Use 3-fold cross-validation
        scoring="accuracy",  # Use accuracy as the measure of model quality
        verbose=2,  # Show detailed progress
    )

    # Fit the model to the data, searching over the defined grid of parameters
    grid_search.fit(X, y)

    # Output the best parameters and the corresponding score after the search
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score found: ", grid_search.best_score_)


if __name__ == "__main__":
    main()
