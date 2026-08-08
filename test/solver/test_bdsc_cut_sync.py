"""The cuts a bdsc column-generation subproblem prices against must be
exactly the master's.

A subproblem cannot arrive at that set by replaying the master's
additions: the master also ages cuts out and declines them (dominance,
similarity) on the strength of solves the subproblem never performs. So
the master sends its whole set, the subproblem mirrors it wholesale, and
the subproblem itself neither purges nor filters.

The instance is the Caroe and Schultz (1999) one from examples/bdsc/cs.py
with r = 2, cut short after a few master steps — convergence is not the
point here, agreement at every sync is.
"""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bdsc.alg_root_bm import BdScAlgRootBm
from pyodsp.dec.bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo
from pyodsp.dec.bdsc.message import BdScDnMessage
from pyodsp.dec.bdsc.run import BdScRun
from pyodsp.alg.bm import cuts_manager as cuts_manager_module
from pyodsp.alg.bm.cuts_manager import CutsManager, NonPurgingCutsManager
from pyodsp.alg.bm.cuts import CutList, OptimalityCut

R = 2
LOWER_BOUND = 1 / 4 - 1 / 32 / (1 + R / 2)
DELTA = 1 / 32 / (1 + R / 2)


def signature(bm):
    """The cuts a master holds, as comparable values."""
    return sorted(
        (
            round(c.cut.rhs, 9),
            tuple(sorted((j, round(v, 9)) for j, v in c.cut.coeffs.items())),
        )
        for group in bm.get_cuts()
        for c in group
    )


def make_root(solver="appsi_highs"):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(LOWER_BOUND, 1))
    model.obj = pyo.Objective(expr=3 * model.x, sense=pyo.minimize)
    alg = BdScAlgRootBm(PyomoSolver(model, SolverConfig(solver), [model.x]))
    return DecNodeRoot(0, alg, log_level_root=0), alg


def make_leaf(s, solver="appsi_highs", cg_iterations=5):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.NonNegativeReals, bounds=(LOWER_BOUND, 1))
    model.y = pyo.Var(within=pyo.Binary)
    h = DELTA * s if s <= R / 2 else 1 / 4 - DELTA * (s - R / 2)
    model.c1 = pyo.Constraint(expr=-model.y / 2 >= h - model.x)
    model.obj = pyo.Objective(expr=-2 * model.y, sense=pyo.minimize)
    # The column-generation master carries a proximal quadratic term, which
    # appsi_highs cannot solve; ipopt can. Its loop is capped well short of
    # the default to keep these tests quick — the invariant under test holds
    # per sync and does not depend on the column generation converging.
    alg = BdScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig(solver), [model.x]),
        SolverConfig("ipopt"),
        cg_iterations,
    )
    node = DecNodeLeaf(s, alg, log_level_leaf=0)
    node.set_bound(-2.01)
    return node, alg


def build_instance():
    root, root_alg = make_root()
    nodes = [root]
    leaf_algs = []
    group = []
    for s in range(1, R + 1):
        leaf, leaf_alg = make_leaf(s)
        nodes.append(leaf)
        leaf_algs.append(leaf_alg)
        root.add_child(s, multiplier=1 / R)
        group.append(s)
    root.set_groups([group])
    return nodes, root_alg, leaf_algs


def run_instrumented(tmp_path, monkeypatch, max_iteration=4, max_cut_age=None):
    """Run bdsc, recording every master/subproblem comparison it allows."""
    if max_cut_age is not None:
        monkeypatch.setattr(cuts_manager_module, "BM_MAX_CUT_AGE", max_cut_age)

    nodes, root_alg, leaf_algs = build_instance()
    record = {
        "mismatches": [],
        "checks": 0,
        "resyncs": 0,
        "purged": 0,
        "snapshots": [],
    }

    original_purge = CutsManager.purge

    def counting_purge(self, model):
        before = self.get_num_cuts()
        original_purge(self, model)
        record["purged"] += before - self.get_num_cuts()

    monkeypatch.setattr(CutsManager, "purge", counting_purge)

    original_pass = BdScAlgLeafPyomo.pass_dn_message
    original_up = BdScAlgLeafPyomo.get_up_message

    def compare(alg):
        record["checks"] += 1
        master, cgsp = signature(root_alg.bm), signature(alg.cgsp)
        if master != cgsp:
            record["mismatches"].append((alg.idx, master, cgsp))

    def patched_pass(self, message):
        original_pass(self, message)
        if message.get_cut_list() is not None:
            record["resyncs"] += 1
            record["snapshots"].append(set(signature(root_alg.bm)))
        compare(self)

    def patched_up(self):
        result = original_up(self)
        compare(self)
        return result

    monkeypatch.setattr(BdScAlgLeafPyomo, "pass_dn_message", patched_pass)
    monkeypatch.setattr(BdScAlgLeafPyomo, "get_up_message", patched_up)

    BdScRun(nodes, Path(tmp_path), level=50, max_iteration=max_iteration).run()
    return record, root_alg, leaf_algs


def test_subproblem_cuts_match_the_master_throughout_a_run(tmp_path, monkeypatch):
    record, _, _ = run_instrumented(tmp_path, monkeypatch)

    assert record["mismatches"] == []
    assert record["checks"] > 0
    assert record["resyncs"] > 0


def test_subproblem_cuts_match_the_master_when_the_master_purges(
    tmp_path, monkeypatch
):
    # BM_MAX_CUT_AGE=1 makes the master drop cuts aggressively — the case a
    # subproblem replaying only the additions gets wrong
    record, _, _ = run_instrumented(
        tmp_path, monkeypatch, max_iteration=12, max_cut_age=1
    )

    # the run has to be long enough that the master is seen to *lose* a cut
    # between two syncs, otherwise every snapshot is a pure addition, which
    # replaying increments would reproduce and this check would pass for free
    dropped = [
        before - after
        for before, after in zip(record["snapshots"], record["snapshots"][1:])
        if before - after
    ]
    assert dropped, "master never dropped a cut between syncs; the check is vacuous"
    assert record["purged"] > 0
    assert record["mismatches"] == []


def test_subproblem_never_purges_on_its_own(tmp_path, monkeypatch):
    _, root_alg, leaf_algs = run_instrumented(tmp_path, monkeypatch, max_cut_age=1)


    assert isinstance(root_alg.bm.cpm.cuts_manager, CutsManager)
    for alg in leaf_algs:
        assert isinstance(alg.cgsp.cpm.cuts_manager, NonPurgingCutsManager)
        assert alg.cgsp.cpm.force is True


def test_dn_message_carries_the_masters_whole_set_not_the_increment(
    tmp_path, monkeypatch
):
    sent = []
    original_pass = BdScAlgLeafPyomo.pass_dn_message

    def patched_pass(self, message):
        sent.append(message.get_cut_list())
        original_pass(self, message)

    monkeypatch.setattr(BdScAlgLeafPyomo, "pass_dn_message", patched_pass)
    nodes, root_alg, _ = build_instance()
    BdScRun(nodes, Path(tmp_path), level=50, max_iteration=4).run()

    snapshots = [len(cut_list[0]) for cut_list in sent if cut_list is not None]
    assert snapshots, "the master never sent its cuts"
    # a snapshot grows with the master's set; an increment would be at most
    # one cut per message
    assert max(snapshots) == len(signature(root_alg.bm))
    assert snapshots == sorted(snapshots)


def test_subproblem_rejects_a_master_with_more_than_one_group():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-100, 100))
    model.s = pyo.Var(domain=pyo.Reals, bounds=(0, 100))
    model.obj = pyo.Objective(expr=model.s, sense=pyo.minimize)
    model.con = pyo.Constraint(expr=model.s >= model.x)
    leaf = BdScAlgLeafPyomo(
        PyomoSolver(model, SolverConfig("appsi_highs"), [model.x]),
        SolverConfig("ipopt"),
    )
    leaf.set_logger(idx=0, depth=1, level=0)
    leaf.build()

    two_groups = [
        CutList([OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)]),
        CutList([OptimalityCut(coeffs={0: 2.0}, rhs=2.0, info={}, objective_value=2.0)]),
    ]

    with pytest.raises(ValueError, match="single group"):
        leaf.pass_dn_message(
            BdScDnMessage(
                solution=[1.0],
                rho=1.0,
                cut_list=two_groups,
                subobj_bounds=[-1e9],
                objective=0.0,
            )
        )
