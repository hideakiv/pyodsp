"""Build-level behaviour of the multistage pipeline.

Nothing here runs SDDP; build() constructs the node lattice without
solving anything.
"""

import logging

import pyomo.environ as pyo
import pytest

from pyodsp.dec.node.dec_node import DecNodeInner, DecNodeLeaf, DecNodeRoot
from pyodsp.model.msp import MultistageProgram
from pyodsp.model.msp.builders import RECEIVED_PREFIX

REALIZATIONS = [
    {"name": "low", "probability": 0.25, "demand": 10.0},
    {"name": "high", "probability": 0.75, "demand": 30.0},
]


def make(*, stages=3, state=("inventory",), **kwargs):
    kwargs.setdefault("log_level", logging.CRITICAL)
    msp = MultistageProgram("t", stage_bound=0.0, **kwargs)

    @msp.stage(state=list(state))
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 100))
        m.buy = pyo.Var(bounds=(0, 100))
        m.balance = pyo.Constraint(
            expr=m.inventory == incoming.inventory + m.buy - node.get("demand", 0.0)
        )
        return 2.0 * m.buy

    msp.set_initial_state(inventory=5.0)
    msp.set_realizations(REALIZATIONS, stages=stages)
    return msp


# -- the shape of the lattice ----------------------------------------------


def test_each_stage_gets_the_node_kind_sddp_expects():
    built = make(stages=4).build()

    assert len(built.nodes) == 4
    assert len(built.nodes[0]) == 1
    assert isinstance(built.nodes[0][0], DecNodeRoot)
    assert all(isinstance(n, DecNodeInner) for n in built.nodes[1])
    assert all(isinstance(n, DecNodeInner) for n in built.nodes[2])
    assert all(isinstance(n, DecNodeLeaf) for n in built.nodes[3])


def test_node_ids_follow_the_stage_index_convention():
    built = make(stages=3).build()

    assert [n.get_idx() for n in built.nodes[0]] == ["0-0"]
    assert [n.get_idx() for n in built.nodes[1]] == ["1-0", "1-1"]


def test_every_node_leads_to_every_node_of_the_next_stage():
    # The lattice recombines — that is what keeps it linear in the horizon.
    built = make(stages=3).build()

    for node in built.nodes[1]:
        assert node.get_children() == ["2-0", "2-1"]
    assert built.nodes[0][0].get_children() == ["1-0", "1-1"]


def test_transition_probabilities_become_the_child_multipliers():
    built = make(stages=3).build()
    root = built.nodes[0][0]

    assert root.get_multiplier("1-0") == pytest.approx(0.25)
    assert root.get_multiplier("1-1") == pytest.approx(0.75)


def test_a_nodes_successors_share_one_cost_to_go():
    built = make(stages=3).build()

    assert built.nodes[0][0].get_groups() == [["1-0", "1-1"]]


# -- the state -------------------------------------------------------------


def test_the_state_is_carried_twice_by_every_middle_stage():
    built = make(stages=3).build()
    model = built.models["1-0"]

    # what it was handed, and what it passes on
    assert model.find_component(f"{RECEIVED_PREFIX}inventory") is not None
    assert model.find_component("inventory") is not None


def test_the_first_stage_receives_no_variable_state():
    built = make(stages=3).build()
    model = built.models["0-0"]

    assert model.find_component(f"{RECEIVED_PREFIX}inventory") is None
    assert "0-0" not in built.subproblems


def test_the_last_stage_passes_nothing_on():
    built = make(stages=3).build()

    assert "2-0" not in built.masters
    assert "2-0" in built.subproblems


def test_consecutive_stages_couple_position_by_position():
    """The check pyodsp.dec cannot make for itself.

    A parent's coupling_dn and a child's coupling_up are matched by
    position with no name check, so they have to be built from one order.
    """
    built = make(stages=3).build()

    passed_on = [v.name for v in built.masters["0-0"].get_vars()]
    received = [v.name for v in built.subproblems["1-0"].get_vars()]

    assert len(passed_on) == len(received) == len(built.labels)
    assert received == [f"{RECEIVED_PREFIX}{name}" for name in passed_on]


def test_an_indexed_state_keeps_its_order_across_the_coupling():
    msp = MultistageProgram("indexed", stage_bound=0.0, log_level=logging.CRITICAL)

    @msp.stage(state=["level"])
    def stage(m, incoming, node):
        m.level = pyo.Var([2, 1], bounds=(0, 10))
        m.c = pyo.Constraint(
            expr=m.level[2] >= incoming.level[2] - node.get("demand", 0.0)
        )
        return sum(m.level[i] for i in [1, 2])

    msp.set_initial_state(level={1: 0.0, 2: 0.0})
    msp.set_realizations(REALIZATIONS, stages=3)
    built = msp.build()

    # declaration order, not sorted
    assert built.labels == ["level[2]", "level[1]"]
    passed_on = [v.name for v in built.masters["0-0"].get_vars()]
    received = [v.name for v in built.subproblems["1-0"].get_vars()]
    assert passed_on == ["level[2]", "level[1]"]
    assert received == [f"{RECEIVED_PREFIX}level[2]", f"{RECEIVED_PREFIX}level[1]"]


# -- the builder contract ---------------------------------------------------


def test_the_state_must_be_named():
    # A stage model holds its own decisions too, so there is nothing
    # sensible to infer.
    msp = MultistageProgram("t", log_level=logging.CRITICAL)

    with pytest.raises(ValueError, match=r"stage\(state=\[\.\.\.\]\) is required"):

        @msp.stage
        def stage(m, incoming, node):
            return 0.0


def test_an_empty_state_leaves_the_stages_unrelated():
    msp = MultistageProgram("t", log_level=logging.CRITICAL)

    with pytest.raises(ValueError, match="leaves the stages uncoupled"):

        @msp.stage(state=[])
        def stage(m, incoming, node):
            return 0.0


def test_a_stage_that_never_declares_the_state_is_told_what_is_missing():
    msp = MultistageProgram("t", stage_bound=0.0, log_level=logging.CRITICAL)

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.buy = pyo.Var(bounds=(0, 10))
        return m.buy

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations(REALIZATIONS, stages=3)

    with pytest.raises(ValueError, match="No variable named 'inventory'"):
        msp.build()


def test_a_missing_initial_value_says_where_to_put_it():
    msp = MultistageProgram("t", stage_bound=0.0, log_level=logging.CRITICAL)

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.inventory >= incoming.inventory)
        return m.inventory

    msp.set_realizations(REALIZATIONS, stages=3)

    with pytest.raises(AttributeError, match="set_initial_state"):
        msp.build()


def test_declaring_an_objective_is_rejected_with_the_alternative():
    msp = MultistageProgram("t", stage_bound=0.0, log_level=logging.CRITICAL)

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 10))
        m.obj = pyo.Objective(expr=m.inventory, sense=pyo.minimize)
        return m.inventory

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations(REALIZATIONS, stages=3)

    with pytest.raises(ValueError, match="the algorithm adds the cost-to-go"):
        msp.build()


def test_a_builder_that_returns_nothing_says_so():
    msp = MultistageProgram("t", stage_bound=0.0, log_level=logging.CRITICAL)

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 10))

    msp.set_initial_state(inventory=0.0)
    msp.set_realizations(REALIZATIONS, stages=3)

    with pytest.raises(ValueError, match="excluding the cost-to-go"):
        msp.build()


def test_missing_declarations_are_reported():
    msp = MultistageProgram("t", log_level=logging.CRITICAL)

    with pytest.raises(ValueError, match="No stage builder"):
        msp.build()

    @msp.stage(state=["inventory"])
    def stage(m, incoming, node):
        m.inventory = pyo.Var(bounds=(0, 10))
        return m.inventory

    with pytest.raises(ValueError, match="No scenario structure"):
        msp.build()


# -- sense ------------------------------------------------------------------


def test_a_maximize_program_reaches_the_algorithms_as_a_minimize_one():
    built = make(stages=3, sense="max").build()

    assert built.masters["0-0"].is_minimize()
    for solver in built.subproblems.values():
        assert solver.is_minimize()


def test_every_node_agrees_on_the_sense_it_was_written_in():
    # An inner node's two solvers share a model, so only the first to be
    # built sees the maximize objective. The node reports through its root
    # algorithm, which is the one that knows.
    built = make(stages=3, sense="max").build()

    assert built.nodes[0][0].get_sense_multiplier() == -1.0
    assert built.nodes[1][0].get_sense_multiplier() == -1.0
    assert built.nodes[2][0].get_sense_multiplier() == -1.0


# -- introspection ----------------------------------------------------------


def test_describe_reports_the_lattice_shape():
    text = make(stages=4).describe()

    assert "stages          : 4" in text
    assert "nodes per stage : [1, 2, 2, 2]" in text
    assert "inventory" in text


# -- MPI --------------------------------------------------------------------


def test_a_serial_program_purges_and_owns_its_results():
    msp = make(stages=3)

    assert msp.rank == 0
    assert msp.is_root_rank
    # cut purging is on, which is the default a single process wants
    assert msp.build().masters["0-0"] is not None


def test_a_replica_rank_does_not_purge_its_cuts(monkeypatch):
    """The one required difference between ranks.

    A replica that aged cuts out on its own schedule would stop matching
    the trial points rank 0 is solving, so only rank 0 may purge.
    """
    from pyodsp.model.msp import problem as problem_module

    msp = make(stages=3, mpi=True)
    monkeypatch.setattr(
        problem_module.MultistageProgram, "rank", property(lambda self: 2)
    )

    assert not msp.is_root_rank
    context = msp._context()
    assert context.purgeable is False


def test_the_root_rank_purges_under_mpi(monkeypatch):
    from pyodsp.model.msp import problem as problem_module

    msp = make(stages=3, mpi=True)
    monkeypatch.setattr(
        problem_module.MultistageProgram, "rank", property(lambda self: 0)
    )

    assert msp.is_root_rank
    assert msp._context().purgeable is True


def test_the_mpi_runner_is_only_imported_when_asked_for():
    # mpi4py is not a hard dependency, so reaching for it on every serial
    # run would make it one.
    from pyodsp.dec.sddp.run import SddpRun

    assert make(stages=3)._runner() is SddpRun
