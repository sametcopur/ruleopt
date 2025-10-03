import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[cnp.float64_t, ndim=1] fill_betas(
    int n,
    double[::1] duals_unique,
    long[::1] inverse_indices,
    float[::1] sample_weight,
    object rng  # Random number generator, keep it generic
):
    cdef:
        double[:] betas = np.zeros(n, dtype=np.float64)
        int[::1] indexes = rng.permutation(np.arange(n, dtype=np.int32))
        float tol = 1e-4  # Precision threshold
        Py_ssize_t i, value_index
        double value, remaining_dual
        float weight

    with nogil:
        for i in indexes:
            value_index = inverse_indices[i]
            remaining_dual = duals_unique[value_index]
            
            if remaining_dual > tol:  # Only process if meaningful dual value remains
                weight = sample_weight[i]
                
                if remaining_dual >= weight:
                    # If enough dual value remains, assign full weight
                    betas[i] = weight
                    duals_unique[value_index] -= weight
                else:
                    # If not enough dual value remains, assign what's left
                    betas[i] = remaining_dual
                    duals_unique[value_index] = 0.0
                    
    return np.asarray(betas)



cpdef solve_ortools(
    double[:, ::1] unique_rows,  # 2D array for constraints
    double[::1] adjusted_sample_weight,  # 1D array for weights
    double normalization_constant,  # Scalar normalization constant
    double penalty,  # Scalar penalty
    str solver_type,
    double[::1] costs  # 1D array for costs
):
    """
    Solves the linear programming problem using OR-Tools in Cython.

    Args:
        unique_rows (np.ndarray): Constraint matrix of size (n_unique, m).
        adjusted_sample_weight (np.ndarray): Weights for the vs variables.
        normalization_constant (float): Normalization constant.
        penalty (float): Penalty term.
        costs (np.ndarray): Cost vector for the ws variables.

    Returns:
        tuple: ws (np.ndarray), duals_unique (np.ndarray)
    """
    from ortools.linear_solver import _pywraplp

    cdef:
        int n_unique = unique_rows.shape[0]
        int m = unique_rows.shape[1]
        int i, j
        double coeff
        double infinity = _pywraplp.Solver_infinity()
        list vs
        list ws

    # Create the linear solver with GLOP backend
    solver = _pywraplp.Solver_CreateSolver(solver_type)

    if solver is None:
        raise RuntimeError("Solver could not be created.")

    # Create variables vs_i >= 0 for i in 0 to n_unique - 1 using vector
    vs = [_pywraplp.Solver_NumVar(solver, 0.0, infinity, f'v_{i}') for i in range(n_unique)]
    ws = [_pywraplp.Solver_NumVar(solver, 0.0, infinity, f'w_{j}') for j in range(m)]

    objective = _pywraplp.Solver_Objective(solver)

    # Add vs variables to the objective
    for i in range(n_unique):
        _pywraplp.Objective_SetCoefficient(objective, vs[i], adjusted_sample_weight[i])

    for j in range(m):
        coeff = normalization_constant * penalty * costs[j]
        _pywraplp.Objective_SetCoefficient(objective, ws[j], coeff)

    _pywraplp.Objective_SetMinimization(objective)

    for i in range(n_unique):
        constraint = _pywraplp.Solver_Constraint(solver, 1.0, infinity)
        _pywraplp.Constraint_SetCoefficient(constraint, vs[i], 1.0)
        for j in range(m):
            coeff = unique_rows[i, j]
            if coeff != 0:
                _pywraplp.Constraint_SetCoefficient(constraint, ws[j], coeff)
    
    _pywraplp.Solver_Solve(solver)

    # Extract solutions using numpy arrays
    cdef cnp.ndarray[double, ndim=1] ws_solution = np.empty(m, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] duals_unique = np.empty(n_unique, dtype=np.float64)

    for j in range(m):
        ws_solution[j] = _pywraplp.Variable_solution_value(ws[j])

    for i in range(n_unique):
        duals_unique[i] = _pywraplp.Constraint_dual_value(_pywraplp.Solver_constraint(solver, i))

    _pywraplp.delete_Solver(solver)

    return ws_solution, duals_unique
