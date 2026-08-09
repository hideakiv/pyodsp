"""The policy an SDDP run leaves behind: a stage's decision for any
incoming state, reconstructable from the saved cuts alone.

Deterministic 3-stage inventory problem (one child per stage), so the
converged decisions are hand-checkable:

    stage 0: buy x0 at cost 1        -> next inventory x0
    stage 1: buy b1 at cost 2, demand 1 -> next inventory prev + b1 - 1
    stage 2: buy b2 at cost 3, demand 2, must cover prev + b2 >= 2

Buying early is strictly cheapest, so the optimum is x0 = 3, b1 = b2 = 0
at a total cost of 3.
"""

import logging
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import pytest

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf, DecNodeInner
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.sddp.run import SddpRun
from pyodsp.dec.sddp.policy import SddpPolicy
from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig

SOLVER = "appsi_highs"


def build_nodes():
    return [[make_root()], [make_inner()], [make_leaf()]]


def make_root():
    model = pyo.ConcreteModel()
    model.next_inventory = pyo.Var(bounds=(0, 10))
    model.obj = pyo.Objective(expr=1 * model.next_inventory, sense=pyo.minimize)
    solver = PyomoSolver(model, SolverConfig(SOLVER), [model.next_inventory])
    node = DecNodeRoot(0, BdAlgRootBm(solver), log_level_root=0)
    node.add_child(1, multiplier=1.0)
    return node


def make_inner():
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    model.buy = pyo.Var(bounds=(0, 10))
    model.next_inventory = pyo.Var(bounds=(0, 10))
    model.balance = pyo.Constraint(
        expr=model.next_inventory == model.prev_inventory + model.buy - 1
    )
    model.obj = pyo.Objective(expr=2 * model.buy, sense=pyo.minimize)
    config = SolverConfig(SOLVER)
    alg_root = BdAlgRootBm(
        PyomoSolver(model, config, [model.next_inventory]), max_iteration=1
    )
    alg_leaf = BdAlgLeafPyomo(PyomoSolver(model, config, [model.prev_inventory]))
    node = DecNodeInner(1, alg_root, alg_leaf, log_level_root=0)
    node.set_bound(0)
    node.add_child(2, multiplier=1.0)
    return node


def make_leaf():
    model = pyo.ConcreteModel()
    model.prev_inventory = pyo.Var()
    model.buy = pyo.Var(bounds=(0, 10))
    model.demand = pyo.Constraint(expr=model.prev_inventory + model.buy >= 2)
    model.obj = pyo.Objective(expr=3 * model.buy, sense=pyo.minimize)
    solver = PyomoSolver(model, SolverConfig(SOLVER), [model.prev_inventory])
    node = DecNodeLeaf(2, BdAlgLeafPyomo(solver))
    node.set_bound(0)
    return node


def run_sddp(filedir):
    nodes = build_nodes()
    SddpRun(
        nodes,
        Path(filedir),
        level=logging.WARNING,
        max_iteration=20,
        sample_frequency=5,
        sample_size=5,
    ).run()
    return nodes


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    filedir = tmp_path_factory.mktemp("sddp_policy")
    nodes = run_sddp(filedir)
    return nodes, filedir


def test_run_converges_to_the_known_optimum(completed_run):
    nodes, _ = completed_run

    assert nodes[0][0].alg_root.bm.obj_bound[-1] == pytest.approx(3.0, abs=1e-6)


def test_policy_reproduces_the_optimal_trajectory(completed_run):
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    trajectory = policy.simulate([0, 1, 2])

    assert [s.node_idx for s in trajectory] == [0, 1, 2]
    assert trajectory[0].next_state[0] == pytest.approx(3.0, abs=1e-6)
    assert trajectory[1].next_state[0] == pytest.approx(2.0, abs=1e-6)
    assert sum(s.stage_cost for s in trajectory) == pytest.approx(3.0, abs=1e-6)


def test_future_cost_is_the_cut_approximation_of_the_cost_to_go(completed_run):
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    solution = policy.evaluate(0)

    assert solution.total_cost == pytest.approx(
        solution.stage_cost + solution.future_cost
    )
    assert solution.future_cost == pytest.approx(
        sum(nodes[0][0].alg_root.bm.get_theta_value())
    )


def test_last_stage_responds_to_the_state_it_is_given(completed_run):
    # the whole point: the saved policy answers for states other than
    # whichever one the last forward pass happened to leave behind
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    assert policy.evaluate(2, [2.0]).stage_cost == pytest.approx(0.0, abs=1e-6)
    assert policy.evaluate(2, [0.0]).stage_cost == pytest.approx(6.0, abs=1e-6)
    assert policy.evaluate(2, [1.0]).stage_cost == pytest.approx(3.0, abs=1e-6)


def test_evaluate_does_not_depend_on_call_order(completed_run):
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    first = policy.evaluate(1, [3.0])
    policy.evaluate(1, [0.0])
    again = policy.evaluate(1, [3.0])

    assert again.next_state == pytest.approx(first.next_state)
    assert again.stage_cost == pytest.approx(first.stage_cost)


def test_inner_stage_solution_is_internally_consistent_at_an_unvisited_state(
    completed_run,
):
    # at a state the run never visited, the cut approximation may price the
    # future imperfectly, so assert the stage's own arithmetic rather than
    # global optimality: balance holds and the cost matches what it bought
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    solution = policy.evaluate(1, [0.5])

    bought = solution.solution["buy"]
    assert solution.next_state[0] == pytest.approx(0.5 + bought - 1)
    assert solution.stage_cost == pytest.approx(2 * bought)


def test_saved_output_holds_cuts_and_not_a_state_conditioned_model(completed_run):
    nodes, filedir = completed_run

    for node_idx in (0, 1):
        node_dir = Path(filedir) / f"node{node_idx}"
        assert (node_dir / "cuts.csv").exists()
        assert not (node_dir / "model.lp").exists()

    saved = pd.read_csv(Path(filedir) / "node0" / "cuts.csv")
    active = [c for group in nodes[0][0].alg_root.bm.get_cuts() for c in group]
    assert len(saved) == len(active)
    assert set(saved["type"]) <= {"optimality", "feasibility"}


def test_policy_restored_from_disk_makes_the_same_decisions(completed_run):
    nodes, filedir = completed_run
    original = SddpPolicy(nodes)
    restored = SddpPolicy.from_saved(build_nodes(), Path(filedir))

    assert restored.evaluate(0).next_state == pytest.approx(
        original.evaluate(0).next_state
    )
    for state in ([3.0], [1.0], [0.0]):
        assert restored.evaluate(1, state).stage_cost == pytest.approx(
            original.evaluate(1, state).stage_cost
        )
        assert restored.evaluate(1, state).next_state == pytest.approx(
            original.evaluate(1, state).next_state
        )


def test_restored_master_holds_the_same_cuts(completed_run):
    nodes, filedir = completed_run
    restored = SddpPolicy.from_saved(build_nodes(), Path(filedir))

    for node_idx in (0, 1):
        original_cuts = [
            c.cut for group in nodes[node_idx][0].alg_root.bm.get_cuts() for c in group
        ]
        restored_cuts = [
            c.cut
            for group in restored.nodes[node_idx].alg_root.bm.get_cuts()
            for c in group
        ]
        assert len(restored_cuts) == len(original_cuts)
        assert [c.coeffs for c in restored_cuts] == [c.coeffs for c in original_cuts]
        assert [c.rhs for c in restored_cuts] == [c.rhs for c in original_cuts]


def test_evaluate_rejects_a_missing_or_mismatched_state(completed_run):
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    with pytest.raises(ValueError, match="prev_state is required"):
        policy.evaluate(2)
    with pytest.raises(ValueError, match="no previous stage"):
        policy.evaluate(0, [1.0])
    with pytest.raises(ValueError, match="couples on 1 variable"):
        policy.evaluate(2, [1.0, 2.0])
    with pytest.raises(ValueError, match="No node with idx"):
        policy.evaluate(99, [1.0])


def test_simulate_rejects_a_path_that_is_not_a_lattice_path(completed_run):
    nodes, _ = completed_run
    policy = SddpPolicy(nodes)

    with pytest.raises(ValueError, match="not a child of"):
        policy.simulate([0, 2])
    with pytest.raises(ValueError, match="passes on no state"):
        policy.simulate([0, 1, 2, 2])
    with pytest.raises(ValueError, match="path is empty"):
        policy.simulate([])
