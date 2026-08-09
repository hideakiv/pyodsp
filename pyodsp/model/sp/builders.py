"""Turning two builder functions into a pyodsp.dec node graph.

Each `build_*` here owns one algorithm's wiring conventions. They differ
more than the shared front-end suggests:

- Benders gives each scenario a *free* copy of the state, which the
  algorithm fixes to the master's trial point. The recourse model carries
  the recourse objective only; the first stage lives in the master.
- Benders with scaled cuts prices columns over the scenario's own
  feasible set, so each scenario needs the real first stage inside it —
  variables, domains and constraints — but still only the recourse
  objective (see examples/bdsc/cs.py).
- Dual decomposition has no master model at all. Every scenario carries a
  complete copy of the problem with its objective scaled by the scenario
  probability, and a coupling model ties the copies together with
  non-anticipativity constraints (see examples/sslp/dd.py).
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.dd.alg_leaf_pyomo import DdAlgLeafPyomo
from pyodsp.dec.dd.alg_root_bm import DdAlgRootBm
from pyodsp.dec.dd.mip_heuristic_root import MipHeuristicRoot
from pyodsp.dec.node.dec_node import DecNodeLeaf, DecNodeRoot
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from .scenario import Scenario, ScenarioSet
from .state import (
    StateSpec,
    StateView,
    add_state_link,
    appears_in_objective,
    domain_and_bounds,
    flatten,
    integer_variables,
    relax_integer_variables,
    replicate_free,
    state_labels,
)

ROOT_IDX = 0

FirstStageBuilder = Callable[[pyo.ConcreteModel], object]
RecourseBuilder = Callable[[pyo.ConcreteModel, StateView, Scenario], object]


@dataclass
class BuildContext:
    """Everything the builders need, resolved and validated."""

    name: str
    first_stage_fn: FirstStageBuilder
    recourse_fn: RecourseBuilder
    specs: List[StateSpec]
    # The first-stage model built once up front. Benders and BDSC adopt it
    # as their master; dual decomposition only reads domains off it.
    reference_model: pyo.ConcreteModel
    reference_expr: object
    scenarios: ScenarioSet
    solver_config: SolverConfig
    cut_master_config: SolverConfig
    is_maximize: bool
    max_iteration: int
    log_level: int
    recourse_bound: float | None
    auto_bound: bool
    relax_recourse: bool
    heuristic: bool


@dataclass
class BuiltProblem:
    """The node graph, plus the handles the result reader needs."""

    method: str
    nodes: List[object]
    root_node: object
    leaf_nodes: List[object] = field(default_factory=list)
    root_solver: PyomoSolver | None = None
    leaf_solvers: Dict[str, PyomoSolver] = field(default_factory=dict)
    leaf_indices: Dict[str, int] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)
    coupling_model: pyo.ConcreteModel | None = None
    relaxed_variables: Dict[str, List[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------


def _reject_user_objective(model: pyo.ConcreteModel, where: str) -> None:
    for objective in model.component_objects(pyo.Objective, active=True):
        raise ValueError(
            f"The {where} builder declared an Objective ({objective.name!r}). "
            "Return the objective expression instead and let the pipeline "
            "place it — it has to convert the sense for the algorithms."
        )


def _set_objective(model: pyo.ConcreteModel, expr, is_maximize: bool) -> None:
    """Install `expr` as the model's objective, in the user's own sense.

    No conversion happens here. PyomoSolver converts a maximize model
    when the solver is built, and remembers the flip so results come back
    in these units — so this stays the one place the user's sense is
    expressed, and the models keep the shape they were described with.
    """
    sense = pyo.maximize if is_maximize else pyo.minimize
    model.add_component("objective", pyo.Objective(expr=expr, sense=sense))


def _require_expression(expr, where: str, scenario: Scenario | None = None):
    if expr is None:
        subject = f"{where} builder"
        if scenario is not None:
            subject += f" (scenario {scenario.name!r})"
        raise ValueError(
            f"The {subject} returned None. It must return the objective "
            "expression for its stage."
        )
    return expr


def build_first_stage_model(fn: FirstStageBuilder, name: str):
    """Run the user's first-stage builder on a fresh model."""
    model = pyo.ConcreteModel(name=name)
    expr = _require_expression(fn(model), "first-stage")
    _reject_user_objective(model, "first-stage")
    return model, expr


def _build_recourse_model(
    ctx: BuildContext, scenario: Scenario, *, embed_first_stage: bool
):
    """Assemble one scenario's model.

    Returns the model, its recourse objective expression, the first-stage
    expression (None unless embedded) and the state vector in canonical
    order.
    """
    model = pyo.ConcreteModel(name=f"{ctx.name}_{scenario.name}")
    if embed_first_stage:
        first_expr = _require_expression(ctx.first_stage_fn(model), "first-stage")
        _reject_user_objective(model, "first-stage")
        state = StateView(model, ctx.specs)
    else:
        first_expr = None
        state = replicate_free(model, ctx.specs)

    before = _state_component_ids(model, ctx.specs)
    recourse_expr = _require_expression(
        ctx.recourse_fn(model, state, scenario), "recourse", scenario
    )
    _reject_user_objective(model, "recourse")
    _reject_state_replacement(model, ctx.specs, before, scenario)
    return model, recourse_expr, first_expr, flatten(model, ctx.specs)


def _state_component_ids(
    model: pyo.ConcreteModel, specs: Sequence[StateSpec]
) -> Dict[str, int]:
    return {spec.name: id(model.find_component(spec.name)) for spec in specs}


def _reject_state_replacement(
    model: pyo.ConcreteModel,
    specs: Sequence[StateSpec],
    before: Dict[str, int],
    scenario: Scenario,
) -> None:
    """Catch a recourse builder that overwrote a state variable.

    Assigning `m.x = Var()` where x is already a component makes Pyomo
    swap the old one out with only a warning. The state variable the
    recourse constraints were meant to reference would then be a fresh,
    uncoupled variable, and the decomposition would optimize the wrong
    problem while looking perfectly healthy.
    """
    replaced = [
        name
        for name, identity in before.items()
        if id(model.find_component(name)) != identity
    ]
    if replaced:
        raise ValueError(
            f"The recourse builder (scenario {scenario.name!r}) replaced the "
            f"state variable(s) {sorted(replaced)}. That name collides with a "
            "first-stage variable, and Pyomo silently swapped the coupled one "
            "out. Rename the recourse variable, and reach the first-stage "
            f"value through `state.{sorted(replaced)[0]}`."
        )


def _coupling_vars_for_benders(model: pyo.ConcreteModel, state_vars: Sequence) -> List:
    """The variables to hand Benders as this scenario's coupling vector.

    Normally the state replicas themselves. When the recourse objective
    touches the state, they are routed through defining constraints first
    — see state.add_state_link for why the cut is otherwise wrong.
    """
    if appears_in_objective(model, state_vars):
        return add_state_link(model, state_vars)
    return list(state_vars)


def _estimate_recourse_bound(ctx: BuildContext, scenario: Scenario) -> float | None:
    """A valid bound on this scenario's recourse objective.

    Computed by minimizing the recourse objective over the *whole*
    problem — first stage included — so the answer bounds the recourse
    value at every trial point the master can produce. Bounding over the
    free state replicas instead would be unbounded for most problems,
    since it is the first-stage constraints that pin the state down.

    Returns None when no finite bound comes back; the master then runs
    with an unbounded theta, which is legal but can leave the first
    iteration unbounded.
    """
    model, recourse_expr, _, _ = _build_recourse_model(
        ctx, scenario, embed_first_stage=True
    )
    if ctx.relax_recourse:
        relax_integer_variables(model)
    _set_objective(model, recourse_expr, ctx.is_maximize)

    solver = pyo.SolverFactory(ctx.solver_config.solver_name)
    try:
        results = solver.solve(model, load_solutions=False, **ctx.solver_config.kwargs)
    except Exception as exc:  # pragma: no cover - solver-specific failures
        warnings.warn(
            f"Could not estimate a recourse bound for scenario "
            f"{scenario.name!r}: {exc}. Continuing without one.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if results.solver.termination_condition != TerminationCondition.optimal:
        warnings.warn(
            f"Could not estimate a recourse bound for scenario "
            f"{scenario.name!r}: the bounding problem terminated with "
            f"{results.solver.termination_condition}. Continuing without one. "
            "Pass recourse_bound=... if the first master turns out unbounded.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    model.solutions.load_from(results)
    return pyo.value(model.objective)


def _recourse_bound(ctx: BuildContext, scenario: Scenario) -> float | None:
    """The value for DecNodeLeaf.set_bound.

    In the user's own units, which is what set_bound takes — the node
    converts it alongside everything else.
    """
    if ctx.recourse_bound is not None:
        return ctx.recourse_bound
    if not ctx.auto_bound:
        return None
    return _estimate_recourse_bound(ctx, scenario)


@dataclass(frozen=True)
class IntegerScan:
    """Where integrality sits in the problem, which decides the algorithm."""

    first_stage: List[str]
    recourse: List[str]

    @property
    def any(self) -> bool:
        return bool(self.first_stage or self.recourse)


def scan_integers(ctx: BuildContext) -> IntegerScan:
    """Locate integer variables, by stage.

    The recourse is probed on the first scenario only — integrality is
    declared by the builder, which is one function, so it does not vary
    with the data in any normal model. build_bd re-checks every scenario
    it actually constructs, so a model that does branch on its data is
    caught there rather than quietly mis-decomposed.
    """
    first_stage = integer_variables(ctx.reference_model)
    probe, _, _, probe_state = _build_recourse_model(
        ctx, ctx.scenarios[0], embed_first_stage=False
    )
    return IntegerScan(first_stage, integer_variables(probe, exclude=probe_state))


# --------------------------------------------------------------------------
# Benders decomposition
# --------------------------------------------------------------------------


def build_bd(ctx: BuildContext) -> BuiltProblem:
    root_model = ctx.reference_model
    _set_objective(root_model, ctx.reference_expr, ctx.is_maximize)

    coupling_dn = flatten(root_model, ctx.specs)
    root_solver = PyomoSolver(root_model, ctx.solver_config, coupling_dn)
    root_alg = BdAlgRootBm(root_solver, max_iteration=ctx.max_iteration)
    root = DecNodeRoot(ROOT_IDX, root_alg, log_level_root=ctx.log_level)

    built = BuiltProblem(
        method="bd",
        nodes=[root],
        root_node=root,
        root_solver=root_solver,
        labels=state_labels(ctx.specs),
    )

    for position, scenario in enumerate(ctx.scenarios):
        idx = position + 1
        model, recourse_expr, _, state_vars = _build_recourse_model(
            ctx, scenario, embed_first_stage=False
        )
        if ctx.relax_recourse:
            built.relaxed_variables[scenario.name] = relax_integer_variables(model)
        _set_objective(model, recourse_expr, ctx.is_maximize)

        # Benders cuts come from LP duals, which an integer recourse does
        # not have. scan_integers only probed one scenario, so this is the
        # check that actually holds for the model being built.
        remaining = integer_variables(model, exclude=state_vars)
        if remaining:
            raise ValueError(
                f"Scenario {scenario.name!r} still has integer recourse "
                f"variables {sorted(set(remaining))[:5]}. Benders cuts are "
                "built from LP duals and would be invalid. Use "
                "method='bd' with integer_recourse='bdsc' (the default), "
                "integer_recourse='relax' to solve the relaxation, or "
                "method='dd'."
            )

        coupling_up = _coupling_vars_for_benders(model, state_vars)
        leaf_solver = PyomoSolver(model, ctx.solver_config, coupling_up)
        leaf = DecNodeLeaf(
            idx, BdAlgLeafPyomo(leaf_solver), log_level_leaf=ctx.log_level
        )
        bound = _recourse_bound(ctx, scenario)
        if bound is not None:
            leaf.set_bound(bound)

        root.add_child(idx, multiplier=scenario.probability)
        built.nodes.append(leaf)
        built.leaf_nodes.append(leaf)
        built.leaf_solvers[scenario.name] = leaf_solver
        built.leaf_indices[scenario.name] = idx

    return built


# --------------------------------------------------------------------------
# Benders decomposition with scaled cuts
# --------------------------------------------------------------------------


def build_bdsc(ctx: BuildContext) -> BuiltProblem:
    root_model = ctx.reference_model
    _set_objective(root_model, ctx.reference_expr, ctx.is_maximize)

    coupling_dn = flatten(root_model, ctx.specs)
    root_solver = PyomoSolver(root_model, ctx.solver_config, coupling_dn)
    root_alg = BdScAlgRootBm(root_solver, max_iteration=ctx.max_iteration)
    root = DecNodeRoot(ROOT_IDX, root_alg, log_level_root=ctx.log_level)

    built = BuiltProblem(
        method="bdsc",
        nodes=[root],
        root_node=root,
        root_solver=root_solver,
        labels=state_labels(ctx.specs),
    )

    group: List[int] = []
    for position, scenario in enumerate(ctx.scenarios):
        idx = position + 1
        # The first stage is embedded, integrality and all: the columns
        # this leaf generates are points of its own feasible set, so that
        # set has to be the real one.
        model, recourse_expr, _, state_vars = _build_recourse_model(
            ctx, scenario, embed_first_stage=True
        )
        _set_objective(model, recourse_expr, ctx.is_maximize)

        leaf_solver = PyomoSolver(model, ctx.solver_config, state_vars)
        leaf = DecNodeLeaf(
            idx,
            BdScAlgLeafPyomo(
                leaf_solver, ctx.cut_master_config, max_iteration=ctx.max_iteration
            ),
            log_level_leaf=ctx.log_level,
        )
        bound = _recourse_bound(ctx, scenario)
        if bound is not None:
            leaf.set_bound(bound)

        root.add_child(idx, multiplier=scenario.probability)
        group.append(idx)
        built.nodes.append(leaf)
        built.leaf_nodes.append(leaf)
        built.leaf_solvers[scenario.name] = leaf_solver
        built.leaf_indices[scenario.name] = idx

    # One aggregated cut over all scenarios, as in examples/bdsc/cs.py.
    root.set_groups([group])
    return built


# --------------------------------------------------------------------------
# Dual decomposition
# --------------------------------------------------------------------------


def build_dd(ctx: BuildContext) -> BuiltProblem:
    if len(ctx.scenarios) < 2:
        raise ValueError(
            "Dual decomposition needs at least two scenarios: with one there "
            "is nothing to make non-anticipative, and the Lagrangian master "
            "has no constraints. Solve the deterministic model directly, or "
            "use method='bd'."
        )

    reference = ctx.reference_model
    info = domain_and_bounds(ctx.specs, reference)
    n_states = len(info)
    n_scenarios = len(ctx.scenarios)

    coupling_model = _build_coupling_model(ctx.name, info, n_states, n_scenarios)
    state_component = coupling_model.component("state")
    vars_dn = {
        position + 1: [state_component[j, position] for j in range(n_states)]
        for position in range(n_scenarios)
    }

    heuristic = MipHeuristicRoot(ctx.solver_config) if ctx.heuristic else None
    root_alg = DdAlgRootBm(
        coupling_model,
        ctx.solver_config,
        vars_dn,
        heuristic,
        max_iteration=ctx.max_iteration,
    )
    root = DecNodeRoot(ROOT_IDX, root_alg, log_level_root=ctx.log_level)

    built = BuiltProblem(
        method="dd",
        nodes=[root],
        root_node=root,
        labels=state_labels(ctx.specs),
        coupling_model=coupling_model,
    )

    for position, scenario in enumerate(ctx.scenarios):
        idx = position + 1
        model, recourse_expr, first_expr, state_vars = _build_recourse_model(
            ctx, scenario, embed_first_stage=True
        )
        # Probability is folded into the scenario objective rather than
        # carried as a child multiplier, so the leaf objectives sum to the
        # true expected cost (see examples/sslp/dd.py).
        _set_objective(
            model, scenario.probability * (first_expr + recourse_expr), ctx.is_maximize
        )

        leaf_solver = PyomoSolver(model, ctx.solver_config, state_vars)
        leaf = DecNodeLeaf(
            idx, DdAlgLeafPyomo(leaf_solver), log_level_leaf=ctx.log_level
        )

        root.add_child(idx)
        built.nodes.append(leaf)
        built.leaf_nodes.append(leaf)
        built.leaf_solvers[scenario.name] = leaf_solver
        built.leaf_indices[scenario.name] = idx

    return built


def _build_coupling_model(
    name: str, info: Dict[int, tuple], n_states: int, n_scenarios: int
) -> pyo.ConcreteModel:
    """The non-anticipativity block that dual decomposition relaxes.

    One replica of the state per scenario, keeping the first stage's real
    domain and bounds, tied in a cycle: replica s equals replica s+1, and
    the last equals the first. The cyclic form is what examples/sslp/dd.py
    uses — it is symmetric in the scenarios, at the cost of one redundant
    constraint block.
    """
    model = pyo.ConcreteModel(name=f"{name}_nonanticipativity")
    model.positions = pyo.RangeSet(0, n_states - 1)
    model.scenarios = pyo.RangeSet(0, n_scenarios - 1)

    model.state = pyo.Var(
        model.positions,
        model.scenarios,
        domain=lambda _, j, s: info[j][0],
        bounds=lambda _, j, s: info[j][1],
    )

    def nonanticipativity_rule(m, j, s):
        return m.state[j, s] == m.state[j, (s + 1) % n_scenarios]

    model.nonanticipativity = pyo.Constraint(
        model.positions, model.scenarios, rule=nonanticipativity_rule
    )
    return model


BUILDERS = {"bd": build_bd, "bdsc": build_bdsc, "dd": build_dd}
