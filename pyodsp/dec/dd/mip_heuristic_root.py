from abc import ABC, abstractmethod
from typing import List, Dict

from pyomo.environ import (
    ConcreteModel,
    Var,
    ScalarVar,
    Constraint,
    Objective,
    RangeSet,
    NonNegativeReals,
    minimize,
    maximize,
    value,
)
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.cuts import OptimalityCut
from pyodsp.alg.bm.cuts_manager import CutInfo
from .message import DdFinalDnMessage


class IMipHeuristicRoot(ABC):
    @abstractmethod
    def build(self, **kwargs) -> None:
        pass

    @abstractmethod
    def run(self) -> Dict[int, DdFinalDnMessage]:
        pass


class MipHeuristicRoot(IMipHeuristicRoot):
    def __init__(self, solver_config: SolverConfig):
        self.solver_config = solver_config

    def _create_master(self, model: ConcreteModel, solver_config: SolverConfig):
        for obj in model.component_objects(Objective, active=True):
            raise ValueError("Objective should not be defined in coupling model")
        if self.is_minimize:
            model._dd_obj = Objective(expr=0.0, sense=minimize)
        else:
            model._dd_obj = Objective(expr=0.0, sense=maximize)
        return PyomoSolver(model, solver_config, [])

    def build(self, **kwargs) -> None:
        self.groups: List[List[int]] = kwargs["groups"]
        coupling_model: ConcreteModel = kwargs["coupling_model"]
        self.cuts: List[List[CutInfo]] = kwargs["cuts"]
        self.vars_dn: Dict[int, List[ScalarVar]] = kwargs["vars_dn"]
        self.is_minimize: bool = kwargs["is_minimize"]
        self.master = self._create_master(coupling_model, self.solver_config)
        for cutlist, group in zip(self.cuts, self.groups):
            assert len(group) == 1
            idx = group[0]
            vars = self.vars_dn[idx]

            num_cuts = len(cutlist)
            minkowski_vars = Var(RangeSet(0, num_cuts - 1), domain=NonNegativeReals)
            self.master.model.add_component(f"_vars_{idx}", minkowski_vars)

            def equality_rule(m, i):
                lhs = 0.0
                for j, cutinfo in enumerate(cutlist):
                    lhs += minkowski_vars[j] * cutinfo.cut.info["solution"][i]
                return lhs == vars[i]

            self.master.model.add_component(
                f"_equality_{idx}",
                Constraint(RangeSet(0, len(vars) - 1), rule=equality_rule),
            )

            self.master.model.add_component(
                f"_convexity_{idx}",
                Constraint(
                    expr=sum(
                        minkowski_vars[i]
                        for i in range(num_cuts)
                        if isinstance(cutlist[i].cut, OptimalityCut)
                    )
                    == 1
                ),
            )

            obj = 0.0
            for j, cutinfo in enumerate(cutlist):
                if self.is_minimize:
                    rhs = cutinfo.constraint.upper
                else:
                    rhs = cutinfo.constraint.lower
                assert rhs is not None

                obj += rhs * minkowski_vars[j]

            self.master.model._dd_obj.expr += obj

    def run(self) -> Dict[int, DdFinalDnMessage]:
        self.master.solve()
        if self.master.is_optimal():
            solutions = {}
            for group in self.groups:
                assert len(group) == 1
                idx = group[0]
                solution = [value(var) for var in self.vars_dn[idx]]
                solutions[idx] = DdFinalDnMessage(solution)

            return solutions
        else:
            # TODO: use logging
            print("WARNING: Unable to find heuristic solution.")
            solutions = {}
            for group in self.groups:
                assert len(group) == 1
                idx = group[0]
                solutions[idx] = DdFinalDnMessage(None)

            return solutions
