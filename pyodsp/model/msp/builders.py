"""Turning one stage builder into the node lattice SDDP runs on.

Each stage model carries the state twice: the value it *received*, which
the algorithm fixes to whatever the previous stage decided, and the value
it *passes on*, which is a decision of its own. They share a name — the
builder writes `state.x` for the first and declares `m.x` for the second
— because they are the same quantity at two times.

That gives each node its two coupling lists:

    coupling_up   the received state, matched against the previous
                  stage's coupling_dn by position
    coupling_dn   the state passed on, matched against the next stage's
                  coupling_up the same way

Stage 0 receives nothing (its state is the initial condition, a constant)
and the last stage passes nothing on, so those two are a parent and a
child respectively while every stage between is both.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import pyomo.environ as pyo

from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.node.dec_node import DecNodeInner, DecNodeLeaf, DecNodeRoot
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

from ..state import StateSpec, StateView, flatten, resolve_state_specs, state_labels
from .lattice import LatticeNode, ScenarioLattice

# An inner node's master is re-solved every time its parent hands it a new
# state, so it takes a single step per visit rather than iterating to
# convergence on a state that is about to change (see examples/balance).
INNER_MASTER_ITERATIONS = 1

RECEIVED_PREFIX = "_received_"

StageBuilder = Callable[[pyo.ConcreteModel, Any, LatticeNode], object]


@dataclass
class MspBuildContext:
    """Everything the builder needs, resolved and validated."""

    name: str
    stage_fn: StageBuilder
    state_names: Sequence[str]
    lattice: ScenarioLattice
    initial_state: Dict[str, Any]
    solver_config: SolverConfig
    is_maximize: bool
    log_level: int
    stage_bound: float | None
    validate: bool
    # Under MPI every rank builds the identical lattice and only rank 0
    # may purge cuts; a replica that aged cuts out on its own schedule
    # would diverge from the trial points rank 0 is actually solving.
    purgeable: bool = True
    specs: List[StateSpec] = field(default_factory=list)


@dataclass
class MspBuiltProblem:
    """The stage-indexed node lattice, plus handles for reading results."""

    nodes: List[List[object]] = field(default_factory=list)
    specs: List[StateSpec] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    masters: Dict[str, PyomoSolver] = field(default_factory=dict)
    subproblems: Dict[str, PyomoSolver] = field(default_factory=dict)
    models: Dict[str, pyo.ConcreteModel] = field(default_factory=dict)

    @property
    def root(self):
        return self.nodes[0][0]


class InitialState:
    """The state handed to stage 0 — numbers, not variables.

    Stage 0 is decided before anything is observed, so what it receives is
    the initial condition. Exposing it under the same `state.x` name the
    other stages use is what lets one builder cover every stage: `state.x`
    is a float here and a variable everywhere else, and a Pyomo expression
    accepts either.
    """

    def __init__(self, values: Dict[str, Any], names: Sequence[str]):
        object.__setattr__(self, "_values", dict(values))
        object.__setattr__(self, "_names", list(names))

    def __getattr__(self, name: str):
        names = object.__getattribute__(self, "_names")
        values = object.__getattribute__(self, "_values")
        if names and name not in names:
            raise AttributeError(
                f"{name!r} is not a state variable; the state is {names}"
            )
        if name not in values:
            raise AttributeError(
                f"No initial value for state variable {name!r}. Pass it to "
                "set_initial_state(...) — the first stage has to start "
                "somewhere."
            )
        return values[name]

    def __getitem__(self, name: str):
        return getattr(self, name)

    def __setattr__(self, name, value):
        raise AttributeError("`state` is read-only")

    def __repr__(self) -> str:
        return f"InitialState({object.__getattribute__(self, '_values')!r})"


class ReceivedState(StateView):
    """`state.x` for a stage that received x from the one before it.

    The received copy cannot simply be called `x`: the builder declares
    its own `x` on the same model for the value it passes on, and Pyomo
    would let the second silently replace the first. So the received copy
    is stored under a reserved prefix and only ever reached through here.
    """

    def __init__(self, model: pyo.ConcreteModel, specs: Sequence[StateSpec]):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_names", [spec.name for spec in specs])

    def __getattr__(self, name: str):
        names = object.__getattribute__(self, "_names")
        if name not in names:
            raise AttributeError(
                f"{name!r} is not a state variable; the state is {names}"
            )
        model = object.__getattribute__(self, "_model")
        return model.find_component(f"{RECEIVED_PREFIX}{name}")


def _reject_user_objective(model: pyo.ConcreteModel, node: LatticeNode) -> None:
    for objective in model.component_objects(pyo.Objective, active=True):
        raise ValueError(
            f"The stage builder declared an Objective ({objective.name!r}) at "
            f"node {node.idx}. Return the stage's own cost expression instead "
            "and let the pipeline place it — the algorithm adds the "
            "cost-to-go."
        )


def _set_objective(model: pyo.ConcreteModel, expr, is_maximize: bool) -> None:
    sense = pyo.maximize if is_maximize else pyo.minimize
    model.add_component("objective", pyo.Objective(expr=expr, sense=sense))


def _require_expression(expr, node: LatticeNode):
    if expr is None:
        raise ValueError(
            f"The stage builder returned None at node {node.idx}. It must "
            "return that stage's own cost expression, excluding the "
            "cost-to-go — the algorithm supplies that."
        )
    return expr


def build_reference_stage(ctx: MspBuildContext):
    """Run the builder once at stage 0 to learn the shape of the state.

    Consecutive stages couple on the state, so it has to be the same shape
    throughout. It is resolved once here and every later stage is held to
    it.
    """
    node = ctx.lattice.nodes(0)[0]
    model = pyo.ConcreteModel(name=f"{ctx.name}_{node.idx}")
    state = InitialState(ctx.initial_state, ctx.state_names)
    expr = _require_expression(ctx.stage_fn(model, state, node), node)
    _reject_user_objective(model, node)
    return model, expr, resolve_state_specs(model, list(ctx.state_names))


def _add_received_state(
    model: pyo.ConcreteModel, specs: Sequence[StateSpec]
) -> List[Any]:
    """Declare the state this stage was handed, flat and in order.

    Free variables with no bounds: the algorithm fixes them to whatever
    the previous stage decided, and a domain of its own could reject a
    perfectly legitimate trial point.
    """
    received: List[Any] = []
    for spec in specs:
        if spec.is_indexed:
            component = pyo.Var(spec.indices, domain=pyo.Reals)
        else:
            component = pyo.Var(domain=pyo.Reals)
        model.add_component(f"{RECEIVED_PREFIX}{spec.name}", component)
        received.extend(
            [component] if not spec.is_indexed else [component[i] for i in spec.indices]
        )
    return received


def _outgoing_state(
    model: pyo.ConcreteModel, specs: Sequence[StateSpec], node: LatticeNode
) -> List[Any]:
    """The state this stage passes on — the builder's own variables."""
    missing = [spec.name for spec in specs if model.find_component(spec.name) is None]
    if missing:
        raise ValueError(
            f"Node {node.idx} never declares {missing}. Every stage but the "
            "last has to declare each state variable, because the value it "
            "leaves is what the next stage receives. The value it was handed "
            f"is `state.{missing[0]}`, which is a different quantity."
        )
    return flatten(model, specs)


def _build_stage_model(ctx: MspBuildContext, node: LatticeNode):
    """One node's model, with the state it received already in place."""
    model = pyo.ConcreteModel(name=f"{ctx.name}_{node.idx}")

    if node.is_first:
        state: Any = InitialState(ctx.initial_state, [s.name for s in ctx.specs])
        received: List[Any] = []
    else:
        received = _add_received_state(model, ctx.specs)
        state = ReceivedState(model, ctx.specs)

    expr = _require_expression(ctx.stage_fn(model, state, node), node)
    _reject_user_objective(model, node)

    passed_on = [] if node.is_last else _outgoing_state(model, ctx.specs, node)
    _set_objective(model, expr, ctx.is_maximize)
    return model, received, passed_on


def build(ctx: MspBuildContext) -> MspBuiltProblem:
    """Assemble the whole lattice of nodes, ready for SddpRun."""
    built = MspBuiltProblem(specs=list(ctx.specs), labels=state_labels(ctx.specs))
    lattice = ctx.lattice
    last_stage = lattice.num_stages - 1

    per_stage: List[List[object]] = []
    for stage in range(lattice.num_stages):
        stage_nodes = []
        for node in lattice.nodes(stage):
            model, received, passed_on = _build_stage_model(ctx, node)
            built.models[node.idx] = model

            # Order matters on an inner node, whose two solvers share one
            # model: the first one built converts a maximize model and
            # records the flip, the second then sees a minimize model and
            # records nothing. The master is built first so the node — which
            # reports through its root algorithm — is the one that knows.
            alg_root = None
            if stage < last_stage:
                master = PyomoSolver(model, ctx.solver_config, passed_on)
                alg_root = BdAlgRootBm(
                    master,
                    max_iteration=1000 if node.is_first else INNER_MASTER_ITERATIONS,
                    purgeable=ctx.purgeable,
                )
                built.masters[node.idx] = master

            alg_leaf = None
            if stage > 0:
                subproblem = PyomoSolver(model, ctx.solver_config, received)
                alg_leaf = BdAlgLeafPyomo(subproblem)
                built.subproblems[node.idx] = subproblem

            dec_node = _make_node(node, alg_root, alg_leaf, ctx)
            if stage < last_stage:
                _attach_children(dec_node, lattice, stage, node.index)
            stage_nodes.append(dec_node)
        per_stage.append(stage_nodes)

    built.nodes = per_stage
    return built


def _make_node(node: LatticeNode, alg_root, alg_leaf, ctx: MspBuildContext):
    if node.is_first:
        return DecNodeRoot(node.idx, alg_root, log_level_root=ctx.log_level)
    if node.is_last:
        leaf = DecNodeLeaf(node.idx, alg_leaf, log_level_leaf=ctx.log_level)
        _set_bound(leaf, ctx)
        return leaf
    inner = DecNodeInner(
        node.idx,
        alg_root,
        alg_leaf,
        # An inner master steps once per visit and would otherwise narrate
        # every one of them.
        log_level_root=logging.CRITICAL,
        log_level_leaf=ctx.log_level,
    )
    _set_bound(inner, ctx)
    return inner


def _set_bound(node, ctx: MspBuildContext) -> None:
    if ctx.stage_bound is not None:
        node.set_bound(ctx.stage_bound)


def _attach_children(
    dec_node, lattice: ScenarioLattice, stage: int, index: int
) -> None:
    """Wire a node to every node of the next stage.

    The lattice recombines, so a node's children are all of them. The
    transition probabilities are the child multipliers, and they form one
    group because a node's successors share a single cost-to-go.
    """
    group = []
    for child_index, probability in enumerate(lattice.transitions_from(stage, index)):
        child_idx = f"{stage + 1}-{child_index}"
        dec_node.add_child(child_idx, multiplier=probability)
        group.append(child_idx)
    dec_node.set_groups([group])
