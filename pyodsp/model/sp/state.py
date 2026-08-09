"""First-stage ("state") variable bookkeeping.

The state vector is the only thing the two stages share, and getting it
wrong is the classic way to build a decomposition that runs happily and
returns the wrong answer: pyodsp.dec matches a parent's coupling_dn list
against a child's coupling_up list *by position*, with no name check. So
one module owns the canonical flattening order, and every list handed to
a PyomoSolver here is produced from it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import pyomo.environ as pyo
from pyomo.core.base.var import VarData
from pyomo.repn import generate_standard_repn


@dataclass(frozen=True)
class StateSpec:
    """One first-stage Var component, plus the index order to read it in.

    `indices` is captured from the reference model rather than recomputed
    per replica: a Pyomo Set iterates in insertion order, and pinning the
    list here means every replica is read in that same order even if it
    was constructed from a differently-ordered set.
    """

    name: str
    indices: List[Any]
    is_indexed: bool

    @classmethod
    def from_component(cls, name: str, component) -> "StateSpec":
        if not isinstance(component, pyo.Var) and not hasattr(component, "is_indexed"):
            raise TypeError(
                f"First-stage state {name!r} is a {type(component).__name__}, "
                "not a Var. Only variables can be state."
            )
        if component.is_indexed():
            return cls(name, list(component.keys()), True)
        return cls(name, [None], False)

    def data_objects(self, model: pyo.ConcreteModel) -> List[VarData]:
        """This spec's VarData on `model`, in canonical order."""
        component = model.find_component(self.name)
        if component is None:
            raise ValueError(
                f"State variable {self.name!r} is missing from model "
                f"{model.name!r}. The first-stage builder must declare the "
                "same variables every time it runs."
            )
        if not self.is_indexed:
            return [component]
        missing = [i for i in self.indices if i not in component]
        if missing:
            raise ValueError(
                f"State variable {self.name!r} is missing indices {missing} "
                f"on model {model.name!r}."
            )
        return [component[i] for i in self.indices]


def resolve_state_specs(
    model: pyo.ConcreteModel, names: Sequence[str] | None
) -> List[StateSpec]:
    """Work out which of `model`'s variables make up the state vector.

    Args:
        model: The reference first-stage model.
        names: Component names to use, in the order given. When None,
            every Var declared on the model is state — the right default
            for two-stage programs, where the recourse generally sees the
            whole first-stage decision.
    """
    if names is None:
        specs = [
            StateSpec.from_component(component.local_name, component)
            for component in model.component_objects(pyo.Var, active=True)
        ]
        if not specs:
            raise ValueError(
                "The first-stage model declares no variables, so there is "
                "nothing for the recourse problems to depend on."
            )
        return specs

    specs = []
    for name in names:
        component = model.find_component(name)
        if component is None:
            declared = [v.local_name for v in model.component_objects(pyo.Var)]
            raise ValueError(
                f"No variable named {name!r} on the first-stage model; "
                f"it declares {declared}."
            )
        specs.append(StateSpec.from_component(name, component))
    if not specs:
        raise ValueError("state=[] leaves the recourse problems uncoupled")
    return specs


def flatten(model: pyo.ConcreteModel, specs: Sequence[StateSpec]) -> List[VarData]:
    """The state vector of `model` as a flat list, in canonical order.

    This is *the* ordering: a parent's coupling_dn and every child's
    coupling_up are all built from this function, so they line up by
    construction rather than by the caller's care.
    """
    return [var for spec in specs for var in spec.data_objects(model)]


def state_labels(specs: Sequence[StateSpec]) -> List[str]:
    """Human-readable name per flat position, for results and plots."""
    labels = []
    for spec in specs:
        for index in spec.indices:
            labels.append(spec.name if index is None else f"{spec.name}[{index}]")
    return labels


class StateView:
    """The `state` argument handed to the recourse builder.

    Attribute access returns the *recourse model's own* copy of a
    first-stage variable, so a constraint written as
    `m.sold[c] == scen.yield_[c] * state.acres[c]` refers to a variable
    that lives in the scenario model, which is what the algorithms need.
    """

    def __init__(self, model: pyo.ConcreteModel, specs: Sequence[StateSpec]):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_names", [spec.name for spec in specs])

    def __getattr__(self, name: str):
        if name not in object.__getattribute__(self, "_names"):
            raise AttributeError(
                f"{name!r} is not a state variable; the state vector is "
                f"{object.__getattribute__(self, '_names')}"
            )
        return object.__getattribute__(self, "_model").find_component(name)

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(
            "`state` is read-only: declare recourse variables on the model "
            "argument instead."
        )

    def __iter__(self):
        return iter(object.__getattribute__(self, "_names"))

    def __repr__(self) -> str:
        return f"StateView({object.__getattribute__(self, '_names')!r})"


def replicate_free(model: pyo.ConcreteModel, specs: Sequence[StateSpec]) -> StateView:
    """Give `model` its own free copy of every state variable.

    Free — domain Reals, no bounds — because these are placeholders that
    the algorithm fixes to whatever trial point the master produced. A
    replica that kept an integer domain would reject the fractional trial
    points an LP master legitimately produces.
    """
    for spec in specs:
        if model.find_component(spec.name) is not None:
            raise ValueError(
                f"The recourse model already has a component named "
                f"{spec.name!r}, which collides with the state variable of "
                "the same name."
            )
        if spec.is_indexed:
            component = pyo.Var(spec.indices, domain=pyo.Reals)
        else:
            component = pyo.Var(domain=pyo.Reals)
        model.add_component(spec.name, component)
    return StateView(model, specs)


def add_state_link(
    model: pyo.ConcreteModel, state_vars: Sequence[VarData]
) -> List[VarData]:
    """Route the coupling through explicit `state == coupling` equalities
    and return the coupling variables to hand to the solver.

    Benders cuts here are assembled from the duals of the constraints the
    coupling variables appear in (see dec.utils.get_nonzero_coefficients_
    from_model). A state variable that only ever appears in the recourse
    *objective* therefore contributes nothing to the cut's gradient while
    still moving its right-hand side — a silently invalid cut. Giving that
    variable a defining constraint puts the missing term back where the
    cut generator can see it.
    """
    n = len(state_vars)
    link_index = pyo.RangeSet(0, n - 1)
    model.add_component("_sp_state", pyo.Var(link_index, domain=pyo.Reals))
    coupling = model.component("_sp_state")

    def link_rule(_, j):
        return state_vars[j] - coupling[j] == 0.0

    model.add_component("_sp_state_link", pyo.Constraint(link_index, rule=link_rule))
    return [coupling[j] for j in range(n)]


def appears_in_objective(
    model: pyo.ConcreteModel, variables: Iterable[VarData]
) -> bool:
    """Whether any of `variables` shows up in the model's active objective."""
    targets = {id(var) for var in variables}
    if not targets:
        return False
    for objective in model.component_objects(pyo.Objective, active=True):
        repn = generate_standard_repn(objective.expr, quadratic=True)
        seen = list(repn.linear_vars) + list(repn.nonlinear_vars or [])
        for pair in repn.quadratic_vars or []:
            seen.extend(pair)
        if any(id(var) in targets for var in seen):
            return True
    return False


def integer_variables(
    model: pyo.ConcreteModel, exclude: Iterable[VarData] = ()
) -> List[str]:
    """Names of the model's integer-valued variables, `exclude` aside."""
    excluded = {id(var) for var in exclude}
    names = []
    for component in model.component_objects(pyo.Var, active=True):
        for index in component:
            var = component[index]
            if id(var) in excluded:
                continue
            if var.is_integer() or var.is_binary():
                names.append(var.name)
    return names


def relax_integer_variables(model: pyo.ConcreteModel) -> List[str]:
    """Widen every integer domain on `model` to its continuous hull.

    Returns the names relaxed. Bounds are preserved: a Binary variable
    becomes a [0, 1] continuous one rather than an unbounded real, which
    is what makes this the LP relaxation and not a different problem.
    """
    relaxed = []
    for component in model.component_objects(pyo.Var, active=True):
        for index in component:
            var = component[index]
            if not (var.is_integer() or var.is_binary()):
                continue
            lower, upper = var.bounds
            var.domain = pyo.Reals
            var.setlb(lower)
            var.setub(upper)
            relaxed.append(var.name)
    return relaxed


def domain_and_bounds(specs: Sequence[StateSpec], model: pyo.ConcreteModel) -> Dict:
    """Per-flat-position domain and bounds, read off the reference model.

    Dual decomposition's coupling model needs replicas that keep the real
    first-stage domain (see examples/sslp/dd.py, whose replicas stay
    Binary), so the non-anticipativity constraints tie together variables
    that can actually take the same values.
    """
    info = {}
    for position, var in enumerate(flatten(model, specs)):
        info[position] = (var.domain, var.bounds)
    return info
