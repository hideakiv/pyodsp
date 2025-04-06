from typing import List
from pathlib import Path

import pandas as pd

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.opt import TerminationCondition

from .solver import Solver


class PyomoSolver(Solver):
    """Base class for solvers using Pyomo"""

    def __init__(
        self, model: pyo.ConcreteModel, solver: str, vars: List[pyo.ScalarVar], **kwargs
    ):
        """Initialize the subsolver.

        Args:
            model: The Pyomo model.
            solver: The solver to use.
            vars: The variables in focus
        """
        self.solver = pyo.SolverFactory(solver)
        self.model = model
        self.vars = vars
        self._solver_kwargs = kwargs

        self.original_objective = self._get_objective()

        self._results = None

        self._infeasible_model = None
        self._unbounded_model = None

    def solve(self) -> None:
        """Solve the model."""

        self._results = self.solver.solve(
            self.model, load_solutions=False, **self._solver_kwargs
        )
        if self.is_optimal():
            self.model.solutions.load_from(self._results)

    def _get_objective(self) -> pyo.Objective:
        """Get the objective of the model"""
        for obj in self.model.component_objects(pyo.Objective, active=True):
            # There should be only one objective
            return obj
        else:
            raise ValueError("Objective not found")

    def get_objective_value(self) -> float:
        """Get the objective value of the model"""
        return pyo.value(self._get_objective())

    def get_original_objective_value(self) -> float:
        """Get the objective value of the model"""
        return pyo.value(self.original_objective)
    
    def activate_original_objective(self) -> None:
        """Activate the original objective"""
        current_obj = self._get_objective()
        current_obj.deactivate()
        self.original_objective.activate()

    def is_minimize(self) -> bool:
        """Get the sense of the objective.

        Returns:
            True if the objective is to be minimized, False otherwise.
        """
        return self._get_objective().sense > 0

    def get_solution(self) -> List[float]:
        """Get the solution of the model."""
        return [var.value for var in self.vars]

    def is_optimal(self) -> bool:
        """Returns whether the model is optimal."""
        return (
            self._results.solver.termination_condition == TerminationCondition.optimal
        )

    def get_dual(self, constrs) -> List[float]:
        return [self.model.dual[constr] for constr in constrs]

    def is_infeasible(self) -> bool:
        """Returns whether the model is infeasible."""
        return (
            self._results.solver.termination_condition
            == TerminationCondition.infeasible
        )

    def get_dual_ray(self, constrs) -> List[float]:
        """Get the dual ray from the infeasibile model."""
        if self._infeasible_model is None:
            self._create_infeasible_model()
        for var in self.vars:
            val = pyo.value(var)
            infs_var = self._infeasible_model.find_component(var.name)
            infs_var.fix(val)
        self.solver.solve(
            self._infeasible_model, load_solutions=True, **self._solver_kwargs
        )
        ray = []
        for constr in constrs:
            infs_constr = self._infeasible_model.find_component(constr.name)
            ray.append(self._infeasible_model.dual[infs_constr])
        return ray

    def get_infeasible_model_objective_value(self) -> float:
        return pyo.value(self._infeasible_model._infeasible_obj)

    def _create_infeasible_model(self) -> None:
        self._infeasible_model = self.model.clone()

        new_obj = 0.0
        for constrs in self._infeasible_model.component_objects(pyo.Constraint):
            constr_name = constrs.name
            if constrs.is_indexed():
                indices = list(constrs.keys())
                plus = pyo.Var(indices, domain=pyo.NonNegativeReals)
                minus = pyo.Var(indices, domain=pyo.NonNegativeReals)
                self._infeasible_model.add_component(f"_var_{constr_name}_plus", plus)
                self._infeasible_model.add_component(f"_var_{constr_name}_minus", minus)
                for index in indices:
                    constr = constrs[index]
                    lower = constr.lower
                    upper = constr.upper
                    if lower is not None and upper is not None:
                        constr.set_value(
                            lower <= constr.body + plus[index] - minus[index] <= upper
                        )
                    elif lower is not None:
                        constr.set_value(
                            lower <= constr.body + plus[index] - minus[index]
                        )
                    elif upper is not None:
                        constr.set_value(
                            constr.body + plus[index] - minus[index] <= upper
                        )
                    new_obj += plus[index] + minus[index]
            else:
                plus = pyo.Var(domain=pyo.NonNegativeReals)
                minus = pyo.Var(domain=pyo.NonNegativeReals)
                self._infeasible_model.add_component(f"_var_{constr_name}_plus", plus)
                self._infeasible_model.add_component(f"_var_{constr_name}_minus", minus)
                lower = constrs.lower
                upper = constrs.upper
                if lower is not None and upper is not None:
                    constrs.set_value(lower <= constrs.body + plus - minus <= upper)
                elif lower is not None:
                    constrs.set_value(lower <= constrs.body + plus - minus)
                elif upper is not None:
                    constrs.set_value(constrs.body + plus - minus <= upper)
                new_obj += plus + minus
        sense = 1
        for obj in self._infeasible_model.component_objects(pyo.Objective, active=True):
            sense = obj.sense
            obj.deactivate()

        self._infeasible_model._infeasible_obj = pyo.Objective(
            expr=new_obj, sense=sense
        )

    def is_unbounded(self) -> bool:
        """Returns whether the model is unbounded."""
        return (
            self._results.solver.termination_condition == TerminationCondition.unbounded
        )

    def get_unbd_ray(self) -> List[float]:
        """Get the unbd ray from the unbounded model."""
        if self._unbounded_model is None:
            self._create_unbounded_model()

        if self._unbounded_model.component("_mod_obj") is not None:
            self._unbounded_model.del_component("_mod_obj")

        # Get the _mod_obj from self.model
        mod_obj = self.model.component("_mod_obj")
        if mod_obj is None:
            raise ValueError("Objective '_mod_obj' not found in the model")

        var_coeff_pairs = self._get_variable_coefficient_pairs(mod_obj.expr)

        obj = 0.0
        for var, coeff in var_coeff_pairs:
            unbd_var = self._unbounded_model.find_component(var.name)
            obj += coeff * unbd_var

        # Transfer _mod_obj to self._unbounded_model
        self._unbounded_model._mod_obj = pyo.Objective(expr=obj, sense=mod_obj.sense)

        self.solver.solve(
            self._unbounded_model, load_solutions=True, **self._solver_kwargs
        )
        ray = []
        for var in self.vars:
            unbd_var = self._unbounded_model.find_component(var.name)
            ray.append(unbd_var.value)
        return ray

    def get_unbounded_model_objective_value(self) -> float:
        return pyo.value(self._unbounded_model._mod_obj)

    def _create_unbounded_model(self) -> None:

        self._unbounded_model = self.model.clone()

        for vars in self._unbounded_model.component_objects(pyo.Var):
            if vars.is_indexed():
                indices = list(vars.keys())
                for index in indices:
                    self._change_domain_to_real(vars[index])
                    if vars[index].lb is None:
                        vars[index].setlb(-1)
                    else:
                        vars[index].setlb(0)
                    if vars.ub is None:
                        vars[index].setub(1)
                    else:
                        vars[index].setub(0)
            else:
                self._change_domain_to_real(vars)
                if vars.lb is not None:
                    vars.setlb(0)
                else:
                    vars.setlb(-1)
                if vars.ub is not None:
                    vars.setub(0)
                else:
                    vars.setub(1)

        for constrs in self._unbounded_model.component_objects(pyo.Constraint):
            if constrs.is_indexed():
                indices = list(constrs.keys())
                for index in indices:
                    constr = constrs[index]
                    lower = constr.lower
                    upper = constr.upper
                    if lower is not None and upper is not None:
                        constr.set_value(0.0 <= constr.body <= 0.0)
                    elif lower is not None:
                        constr.set_value(0.0 <= constr.body)
                    elif upper is not None:
                        constr.set_value(constr.body <= 0.0)
            else:
                lower = constrs.lower
                upper = constrs.upper
                if lower is not None and upper is not None:
                    constrs.set_value(0.0 <= constrs.body <= 0.0)
                elif lower is not None:
                    constrs.set_value(0.0 <= constrs.body)
                elif upper is not None:
                    constrs.set_value(constrs.body <= 0.0)

    def _get_variable_coefficient_pairs(self, expr):
        """Get variable-coefficient pairs from a Pyomo expression."""
        repn = generate_standard_repn(expr)
        var_coeff_pairs = list(zip(repn.linear_vars, repn.linear_coefs))
        return var_coeff_pairs

    def save(self, dir: Path) -> None:
        """outputs solution to dir"""
        path = dir / "sol.csv"
        solution = {}
        for v in self.model.component_objects(pyo.Var, active=True):
            for index in v:
                if index is None:
                    varname = str(v)
                else:
                    varname = f"{v}_{index}"
                solution[varname] = v[index].value

        # Convert the solution to a DataFrame
        sol = pd.DataFrame(list(solution.items()), columns=["var", "val"])

        sol.to_csv(path, sep="\t", index=False)

    def _change_domain_to_real(self, var: pyo.ScalarVar) -> None:
        if var.domain is pyo.NonNegativeIntegers:
            var.domain = pyo.NonNegativeReals
        elif var.domain is pyo.Integers:
            var.domain = pyo.Reals
        elif var.domain is pyo.NonPositiveIntegers:
            var.domain = pyo.NonPositiveReals
