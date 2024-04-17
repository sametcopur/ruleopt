"""
Example: Optimizing RUGClassifier Parameters with Optuna on the Iris Dataset

This example demonstrates the use of Optuna for hyperparameter optimization 
of the RUGClassifier on the Iris dataset. It optimizes both the classifier's
parameters and the tree parameters within the `tree_parameters` dictionary
to maximize classification accuracy. The Gini index is used as the rule cost 
function for the rule generation process.
"""

# Import necessary libraries
import optuna
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini, Length
from ruleopt.solver import ORToolsSolver
# Define the objective function for the Optuna study
def objective(trial):
    # Set a random state for reproducibility
    random_state = 42

    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Suggest values for the RUGClassifier's hyperparameters
    penalty = trial.suggest_float("penalty", 1, 10.0)
    max_rmp_calls = trial.suggest_int("max_rmp_calls", 0, 20)

    # Suggest values for the tree parameters
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    class_weight_option = trial.suggest_categorical("class_weight", ["balanced", None])

    cost_function_choice = trial.suggest_categorical(
        "cost_function", ["Gini", "Length"]
    )
    if cost_function_choice == "Gini":
        cost_function = Gini()
    else:
        cost_function = Length()

    tree_parameters = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight_option,
    }

    # Initialize the RUGClassifier with suggested parameters
    solver = ORToolsSolver(penalty=penalty)
    
    rug = RUGClassifier(
        solver,
        random_state=random_state,
        max_rmp_calls=max_rmp_calls,
        rule_cost=cost_function,
        **tree_parameters
    )

    # Fit the RUGClassifier to the training data
    rug.fit(X_train, y_train)

    # Predict the labels of the testing set
    y_pred = rug.predict(X_test)

    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def main():
    # Create an Optuna study object that maximizes the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Print the best parameters and the best accuracy achieved
    print("Best parameters:", study.best_params)
    print(f"Best accuracy: {study.best_value:.2f}")


if __name__ == "__main__":
    main()
