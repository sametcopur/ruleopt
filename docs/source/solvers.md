# Supported Solvers

`ruleopt` is designed to work with a variety of solvers. Here's a detailed overview of the solvers that `ruleopt` supports.

## Gurobi and CPLEX

**ruleopt** also offers integration with high-performance, _proprietary_ solvers like `GurobiSolver` and `CPLEXSolver`. These solvers work with their respective Python interfaces, `gurobipy` for **Gurobi** and `docplex` for **CPLEX**. It is important to note that both solvers require a valid license to use.

## OR-Tools

In addition to the above, `ruleopt` integrates with Google's **OR-Tools**, offering access to a comprehensive suite of solvers. _For users primarily interested in free solvers, we recommend starting with the default configuration provided by OR-Tools for simplicity and ease of use._

For the list of supported solvers through, please see **OR-Tools** [webpage](https://developers.google.com/optimization).

### Note

While **OR-Tools** supports a broad range of solvers, including **Gurobi** and **CPLEX**, setting up some solvers, especially the commercial ones or those disabled by default (**GLPK**, **HiGHS**), requires compiling OR-Tools from source. This process can be more complex but offers flexibility for users who need these specific solvers.