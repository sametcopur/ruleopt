import numpy as np
from typing import Tuple, List
from scipy.sparse import csr_matrix
from ..utils import check_module_available

GUROBI_AVAILABLE = check_module_available("gurobipy")
RAY_AVAILABLE = check_module_available("ray")


def gurobi_solver(
    dist_a_hat: csr_matrix,
    costs: np.array,
    ws_par: np.array,
    theta_k: np.array,
    ws0: np.array = None,
    penalty: int = 1.0,
    learning_rate=0.01,
    *args,
    **kwargs,
) -> Tuple[np.array, np.array, List]:
    """
    Solve optimization problem using Gurobi solver.

    Args:
    _coeffs (Coefficients): The coefficients of the optimization problem.
    k (int): The number of constraints.
    ws0 (list, optional): Initial guess for the ws. Defaults to [].
    vs0 (list, optional): Initial guess for the vs. Defaults to [].
    penalty (int, optional): Penalty parameter. Defaults to 1.0.
    normConst (int, optional): Normalization constant. Defaults to 1.
    *args: Variable length argument list.
    **kwargs: Arbitrary keyword arguments.

    Returns:
    Tuple[np.array, np.array, List]: The optimal ws, vs and dual values.
    """
    import gurobipy as gp
    
    with gp.Env() as env, gp.Model(env=env) as modprimal:

        # Create the Ahat matrix from given coefficients
        n, m = dist_a_hat.shape

        # Set up the primal model
        modprimal.setParam("OutputFlag", False)

        # Define variables
        vs = modprimal.addMVar(shape=int(n), name="vs")
        ws = modprimal.addMVar(shape=int(m), name="ws")

        if ws0 is not None:
            tempws = np.zeros(m)
            tempws[: len(ws0)] = ws0
            ws.setAttr("Start", tempws)

        # Set objective function
        objective = (
            np.ones(n) @ vs
            + (costs * penalty) @ ws
            + theta_k @ (ws - ws_par)
            + (learning_rate / 2) * (ws - ws_par) @ (ws - ws_par)
        )
        modprimal.setObjective(objective, gp.GRB.MINIMIZE)

        # Add constraints
        modprimal.addConstr(dist_a_hat @ ws + vs >= 1.0, name="Ahat Constraints")

        # Optimize the model
        modprimal.update()
        modprimal.optimize()

        # Extract betas from the dual values of the constraints
        betas = np.array(modprimal.getAttr(gp.GRB.Attr.Pi)[:n])

        return ws.X, betas


def gurobi_parallel_solver(
    parallel: bool,
    par_n_dist: int,
    dist_a_hat: csr_matrix,
    costs: np.array,
    ws_par: np.array,
    theta: np.array,
    ws0: np.array = None,
    penalty: int = 2.0,
    learning_rate=0.01,
) -> Tuple[np.array, np.array, List]:

    if not GUROBI_AVAILABLE and not RAY_AVAILABLE:
        raise ImportError(
                "Gurobi and ray are required for this class but is not installed.",
                "Please install it with 'pip install gurobipy'",
                'and pip install "ray[default]"'
            )

    import ray

    if parallel:
        # Ray ile paralel çalıştırmak için işlevi remote olarak işaretle ve çalıştır.
        ray_func = ray.remote(gurobi_solver)
        results = [
            ray_func.remote(
                dist_a_hat=dist_a_hat[dist],
                costs=costs,
                theta_k=theta[dist],
                learning_rate=learning_rate,
                ws_par=ws_par,
                penalty=penalty,
                ws0=None if ws0 is None else ws0[dist],
            )
            for dist in range(par_n_dist)
        ]
        results = ray.get(results)
    else:
        results = list()
        for dist in range(par_n_dist):
            results.append(
                gurobi_solver(
                    dist_a_hat=dist_a_hat[dist],
                    costs=costs,
                    penalty=penalty,
                    theta_k=theta[dist],
                    learning_rate=learning_rate,
                    ws_par=ws_par,
                    ws0=None if ws0 is None else ws0[dist],
                )
            )
    return results
