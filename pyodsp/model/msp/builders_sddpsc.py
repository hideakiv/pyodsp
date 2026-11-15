"""Turning one stage builder into the node lattice sddpsc runs on.

Plain SDDP builds one model per node and lets both of an inner node's
roles share it (see builders.py). SDDP with scaled cuts cannot: each role
runs its own cut-adding bundle method with its own objective, and the leaf
role's column generation needs the incoming state free while the root role
needs it fixed (see pyodsp.dec.sddpsc). So an inner node gets two models,
both from the same stage builder:

    root role   the stage on its own, the state it received held in
                mutable Params that SddpScNodeInner sets on every visit;
                it passes its own state on (coupling_dn), as in SDDP
    leaf role   a chain of Blocks replaying the stage transitions from
                stage t - lookback (stage 0 by default) through t - 1,
                then stage t itself. It couples upward on the state stage
                t - 1 passes on, and carries stage t's outgoing state as
                future_vars, for the next stage's cost-to-go

A last-stage node is only the leaf role, and stage 0 only the root role.

The chain's earlier stages are built with the first node of their stage
and their costs are dropped: only their transitions matter, and sddpsc
requires those to be the same at every node of a stage (only the costs may
be random). With validate on, that is checked by building every node of
every stage and comparing what they constrain.
"""

import logging
from typing import Any, List, Sequence

import pyomo.environ as pyo

from pyodsp.dec.node.dec_node import DecNodeLeaf, DecNodeRoot
from pyodsp.dec.sddpsc.alg_leaf_pyomo import SddpScAlgLeafPyomo
from pyodsp.dec.sddpsc.alg_root_bm import SddpScAlgRootBm
from pyodsp.dec.sddpsc.node import SddpScNodeInner
from pyodsp.solver.pyomo_solver import PyomoSolver

from ..state import StateSpec, StateView, flatten, state_labels
from .builders import (
    RECEIVED_PREFIX,
    InitialState,
    MspBuildContext,
    MspBuiltProblem,
    ReceivedState,
    _add_received_state,
    _attach_children,
    _build_stage_model,
    _outgoing_state,
    _reject_user_objective,
    _require_expression,
    _set_objective,
)
from .lattice import LatticeNode


def build_sddpsc(ctx: MspBuildContext) -> MspBuiltProblem:
    """Assemble the whole lattice of nodes, ready for SddpScRun."""
    if ctx.stage_bound is None:
        raise ValueError(
            "SDDP with scaled cuts needs stage_bound: every child's cost "
            "needs a floor before the column-generation subproblem pricing "
            "it is bounded."
        )
    if ctx.lookback is not None and ctx.lookback < 1:
        raise ValueError(
            f"lookback must be at least 1, got {ctx.lookback}: a leaf couples "
            "on the state the previous stage passes on, so the chain has to "
            "include that stage for the state to have a domain at all"
        )
    if ctx.validate:
        _check_deterministic_transitions(ctx)

    built = MspBuiltProblem(specs=list(ctx.specs), labels=state_labels(ctx.specs))
    lattice = ctx.lattice
    last_stage = lattice.num_stages - 1

    per_stage: List[List[object]] = []
    for stage in range(lattice.num_stages):
        stage_nodes = []
        for node in lattice.nodes(stage):
            if node.is_first:
                dec_node = _make_root(ctx, node, built)
            elif node.is_last:
                dec_node = _make_leaf(ctx, node, built)
            else:
                dec_node = _make_inner(ctx, node, built)
            if stage < last_stage:
                _attach_children(dec_node, lattice, stage, node.index)
            stage_nodes.append(dec_node)
        per_stage.append(stage_nodes)

    built.nodes = per_stage
    return built


def _make_root(ctx: MspBuildContext, node: LatticeNode, built: MspBuiltProblem):
    model, _, passed_on = _build_stage_model(ctx, node)
    master = PyomoSolver(model, ctx.solver_config, passed_on)
    built.models[node.idx] = model
    built.masters[node.idx] = master
    return DecNodeRoot(node.idx, SddpScAlgRootBm(master), log_level_root=ctx.log_level)


def _make_leaf(ctx: MspBuildContext, node: LatticeNode, built: MspBuiltProblem):
    model, coupling_up, _ = _build_chain_model(ctx, node)
    subproblem = PyomoSolver(model, ctx.solver_config, coupling_up)
    built.models[node.idx] = model
    built.subproblems[node.idx] = subproblem
    leaf = DecNodeLeaf(
        node.idx,
        SddpScAlgLeafPyomo(subproblem, ctx.solver_config),
        log_level_leaf=ctx.log_level,
    )
    leaf.set_bound(ctx.stage_bound)
    return leaf


def _make_inner(ctx: MspBuildContext, node: LatticeNode, built: MspBuiltProblem):
    own, params, passed_on = _build_root_role_model(ctx, node)
    master = PyomoSolver(own, ctx.solver_config, passed_on)
    alg_root = SddpScAlgRootBm(master, state_params=params)

    chain, coupling_up, future_vars = _build_chain_model(ctx, node)
    subproblem = PyomoSolver(chain, ctx.solver_config, coupling_up)
    alg_leaf = SddpScAlgLeafPyomo(
        subproblem, ctx.solver_config, future_vars=future_vars
    )

    built.models[node.idx] = own
    built.masters[node.idx] = master
    built.subproblems[node.idx] = subproblem
    inner = SddpScNodeInner(
        node.idx,
        alg_root,
        alg_leaf,
        # as in builders._make_node: an inner master steps once per visit
        # and would otherwise narrate every one of them
        log_level_root=logging.CRITICAL,
        log_level_leaf=ctx.log_level,
    )
    inner.set_bound(ctx.stage_bound)
    return inner


def _add_received_params(block, specs: Sequence[StateSpec]) -> List[Any]:
    """The received state as mutable Params, flat and in canonical order --
    the root role's copy, which SddpScNodeInner sets on every visit."""
    params: List[Any] = []
    for spec in specs:
        if spec.is_indexed:
            component = pyo.Param(spec.indices, mutable=True, initialize=0.0)
        else:
            component = pyo.Param(mutable=True, initialize=0.0)
        block.add_component(f"{RECEIVED_PREFIX}{spec.name}", component)
        params.extend(
            [component] if not spec.is_indexed else [component[i] for i in spec.indices]
        )
    return params


def _build_root_role_model(ctx: MspBuildContext, node: LatticeNode):
    """An inner node's root role: the stage alone, state received as Params."""
    model = pyo.ConcreteModel(name=f"{ctx.name}_{node.idx}")
    params = _add_received_params(model, ctx.specs)
    expr = _require_expression(
        ctx.stage_fn(model, ReceivedState(model, ctx.specs), node), node
    )
    _reject_user_objective(model, node)
    passed_on = _outgoing_state(model, ctx.specs, node)
    _set_objective(model, expr, ctx.is_maximize)
    return model, params, passed_on


def _build_chain_model(ctx: MspBuildContext, node: LatticeNode):
    """A leaf role: stages t - lookback .. t as Blocks, linked by state.

    Returns the model, the state stage t received (coupling_up) -- stage
    t - 1's own outgoing variables, so with whatever bounds the builder
    gave them -- and the state stage t passes on (future_vars; empty at
    the last stage).
    """
    t = node.stage
    start = 0 if ctx.lookback is None else max(0, t - ctx.lookback)
    model = pyo.ConcreteModel(name=f"{ctx.name}_{node.idx}_chain")

    previous = None
    coupling_up: List[Any] = []
    expr = None
    for k in range(start, t + 1):
        block = pyo.Block()
        model.add_component(f"stage_{k}", block)
        at = node if k == t else ctx.lattice.nodes(k)[0]

        if previous is not None:
            state: Any = StateView(previous, ctx.specs)
        elif k == 0:
            state = InitialState(ctx.initial_state, [s.name for s in ctx.specs])
        else:
            # the chain starts after stage 0: what it receives is free
            _add_received_state(block, ctx.specs)
            state = ReceivedState(block, ctx.specs)

        expr = _require_expression(ctx.stage_fn(block, state, at), at)
        _reject_user_objective(block, at)
        if k < t:
            # this stage's cost belongs to its own node; only its
            # transition is kept
            outgoing = _outgoing_state(block, ctx.specs, at)
            if k == t - 1:
                coupling_up = outgoing
        previous = block

    future_vars = [] if node.is_last else _outgoing_state(previous, ctx.specs, node)
    _set_objective(model, expr, ctx.is_maximize)
    return model, coupling_up, future_vars


def _transition_signature(ctx: MspBuildContext, node: LatticeNode):
    """What a node's stage constrains, cost aside, as comparable values."""
    model = pyo.ConcreteModel()
    if node.is_first:
        state: Any = InitialState(ctx.initial_state, [s.name for s in ctx.specs])
    else:
        _add_received_state(model, ctx.specs)
        state = ReceivedState(model, ctx.specs)
    ctx.stage_fn(model, state, node)
    constraints = sorted(
        (c.name, str(c.lower), str(c.body), str(c.upper))
        for c in model.component_data_objects(pyo.Constraint, active=True)
    )
    variables = sorted(
        (v.name, str(v.lb), str(v.ub), str(v.domain))
        for v in model.component_data_objects(pyo.Var)
    )
    return constraints, variables


def _check_deterministic_transitions(ctx: MspBuildContext) -> None:
    lattice = ctx.lattice
    for stage in range(lattice.num_stages):
        nodes = lattice.nodes(stage)
        reference = _transition_signature(ctx, nodes[0])
        for node in nodes[1:]:
            if _transition_signature(ctx, node) != reference:
                raise ValueError(
                    f"Node {node.idx} constrains its stage differently from "
                    f"node {nodes[0].idx}. SDDP with scaled cuts needs every "
                    "node of a stage to share one transition -- the same "
                    "constraints and variable bounds -- with only the cost "
                    "random; see pyodsp.dec.sddpsc. (validate=False skips "
                    "this check.)"
                )
