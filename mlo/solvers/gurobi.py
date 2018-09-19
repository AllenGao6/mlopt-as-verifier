import gurobipy as grb
import numpy as np
from . import statuses as s
from .results import Results
from ..constants import INFINITY, TOL


class GUROBISolver(object):

    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.PRIMAL_INFEASIBLE,
                  5: s.DUAL_INFEASIBLE,
                  4: s.PRIMAL_OR_DUAL_INFEASIBLE,
                  6: s.SOLVER_ERROR,
                  7: s.MAX_ITER_REACHED,
                  8: s.SOLVER_ERROR,
                  9: s.TIME_LIMIT,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, problem):
        '''
        Solve problem

        Args:
            problem (OptimizationProblem): problem to be solved

        Returns:
            Results structure
        '''
        p = problem.data

        if p.A is not None:
            # Convert Matrices in CSR format
            p.A = p.A.tocsr()

        # Get problem dimensions
        m, n = p.A.shape

        # Adjust infinity values in bounds
        u = np.copy(p.u)
        l = np.copy(p.l)

        for i in range(m):
            if u[i] >= INFINITY:
                u[i] = grb.GRB.INFINITY
            if l[i] <= -INFINITY:
                l[i] = -grb.GRB.INFINITY

        # Create a new model
        model = grb.Model("")

        # Add variables
        for i in range(n):
            model.addVar(ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)
        model.update()
        x = model.getVars()

        # Set integer variables
        if problem.is_mip():
            for i in p.int_idx:
                x[i].vtype = grb.INTEGER

        # Add inequality constraints: iterate over the rows of A
        # adding each row into the model
        if p.A is not None:
            for i in range(m):
                start = p.A.indptr[i]
                end = p.A.indptr[i+1]
                variables = [x[j] for j in p.A.indices[start:end]]  # nnz
                coeff = p.A.data[start:end]
                expr = grb.LinExpr(coeff, variables)
                if (np.abs(l[i] - u[i]) < TOL):
                    model.addConstr(expr, grb.GRB.EQUAL, u[i])
                elif (l[i] == -grb.GRB.INFINITY) & (u[i] == grb.GRB.INFINITY):
                    # Dummy constraint that is always satisfied.
                    # Gurobi crashes if both constraints in addRange function
                    # are infinite.
                    model.addConstr(0.*expr, grb.GRB.LESS_EQUAL, 10.)
                else:
                    model.addRange(expr, lower=l[i], upper=u[i])

        # Define objective
        obj = grb.LinExpr(p.c, x)  # Linear part of the objective
        #  if p.P is not None:
        #      if p.P.count_nonzero():  # If there are any nonzero elms in P
        #          for i in range(p.P.nnz):
        #              obj.add(.5*p.P.data[i] *
        #                      x[p.P.row[i]]*x[p.P.col[i]])
        model.setObjective(obj)  # Set objective

        # Set parameters
        if 'verbose' in self._settings:  # if verbose is null, suppress it
            if self._settings['verbose'] == 0:
                model.setParam("OutputFlag", 0)
        else:
                # If verbose not specified, suppress it as well
                model.setParam("OutputFlag", 0)

        for param, value in self._settings.items():  # Set other parameters
            if param != "verbose":
                model.setParam(param, value)

        # Update model
        model.update()

        # Solve problem
        try:
            model.optimize()
        except:  # Error in the solution
            if self._settings['verbose']:
                print("Error in GUROBI solution\n")
            run_time = model.Runtime
            return Results(s.SOLVER_ERROR, None, None, None, run_time, None)

        # Get status
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)

        # Get computation time
        run_time = model.Runtime

        # Total Number of iterations
        niter = model.BarIterCount

        if status in s.SOLUTION_PRESENT:
            # Get objective value
            objval = model.objVal

            # Get solution
            x = np.array([x[i].X for i in range(n)])

            # Get dual variables  (Gurobi uses swapped signs (-1))
            if not problem.is_mip():
                constrs = model.getConstrs()
                y = -np.array([constrs[i].Pi for i in range(m)])
            else:
                y = None

            # Get active constraints
            active_cons = self.active_constraints(model)

            return Results(status, objval, x, y,
                           run_time, niter, active_cons)
        else:
            return Results(status, None, None, None,
                           run_time, niter, None)

    def active_constraints(self, model):

        active_constr = np.zeros(model.NumConstrs, dtype=int)
        basis = model.getAttr(grb.AttrConstClass.CBasis)

        for i in range(model.NumConstrs):
            if basis[i] == -1:
                active_constr[i] = 1  # Only upper bounds

        return active_constr

