"""Objective-sense primitives.

Kept apart from pyomo_utils so PyomoSolver can use them: pyomo_utils
imports PyomoSolver for its other helpers, and importing back the other
way would be a cycle.
"""

from pyomo.environ import ConcreteModel, Objective, maximize, minimize


def model_is_minimize(model: ConcreteModel) -> bool:
    """Whether `model`'s active objective is a minimization.

    Raises:
        ValueError: If the model has no active objective.
    """
    for objective in model.component_objects(Objective, active=True):
        return objective.sense == minimize
    raise ValueError("Objective not found")


def negate_objective_sense(model: ConcreteModel) -> None:
    """Flip `model`'s active Objective between maximize and minimize,
    negating its expression so the optimal solution is unchanged and the
    optimal objective value negates too.

    PyomoSolver applies this on construction, so models reach the
    decomposition algorithms already in the minimize form they require;
    it is exposed here for code that needs the conversion on its own.
    """
    for obj in model.component_objects(Objective, active=True):
        new_sense = minimize if obj.sense == maximize else maximize
        expr = obj.expr
        obj.sense = new_sense
        obj.set_value(-expr)
        return
    raise ValueError("No active objective found on model")
