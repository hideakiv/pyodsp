"""BundleMethod.get_cut_list/replace_cuts — the mechanism LatticeMpi uses to
keep a non-purgeable replica's cuts identical to a purging source's,
without the replica ever re-deciding what's added or stale.
"""

import pyomo.environ as pyo
import pytest

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.alg.bm.bm import BundleMethod
from pyodsp.alg.bm.cuts import CutList, OptimalityCut


def make_bm(purgeable):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals, bounds=(-10, 10))
    model.obj = pyo.Objective(expr=model.x, sense=pyo.minimize)
    solver = PyomoSolver(model, SolverConfig("appsi_highs"), [model.x])
    bm = BundleMethod(solver, force=not purgeable, purgeable=purgeable)
    bm.set_logger(node_id=0, depth=0)
    bm.build(1, [-1e9])
    return bm


def cut_coeffs(bm):
    return [c.cut.coeffs for group in bm.get_cuts() for c in group]


def test_replace_cuts_mirrors_source_snapshot_on_a_fresh_replica():
    source = make_bm(purgeable=True)
    replica = make_bm(purgeable=False)

    source.add_cuts(
        [CutList([OptimalityCut(coeffs={0: 1.0}, rhs=2.0, info={}, objective_value=2.0)])]
    )

    replica.replace_cuts(source.get_cut_list())

    assert cut_coeffs(replica) == cut_coeffs(source)


def test_replace_cuts_bypasses_dominance_check_unlike_add_cuts():
    # A replica's own theta is stale (it never solves the source's trial
    # points), so if replace_cuts naively called add_cuts without force,
    # a cut could be silently skipped as "already dominated" by that
    # stale theta — force=True (set automatically for purgeable=False,
    # see BdAlgRootBm) is what replace_cuts relies on to avoid this.
    source = make_bm(purgeable=True)
    replica = make_bm(purgeable=False)
    assert replica.cpm.force is True

    source.add_cuts(
        [CutList([OptimalityCut(coeffs={0: 1.0}, rhs=2.0, info={}, objective_value=2.0)])]
    )
    replica.replace_cuts(source.get_cut_list())

    source.add_cuts(
        [CutList([OptimalityCut(coeffs={0: 2.0}, rhs=3.0, info={}, objective_value=3.0)])]
    )
    replica.replace_cuts(source.get_cut_list())

    assert cut_coeffs(replica) == cut_coeffs(source)
    assert len(cut_coeffs(replica)) == 2


def test_replace_cuts_wholesale_replaces_not_accumulates():
    source = make_bm(purgeable=True)
    replica = make_bm(purgeable=False)

    source.add_cuts(
        [CutList([OptimalityCut(coeffs={0: 1.0}, rhs=2.0, info={}, objective_value=2.0)])]
    )
    replica.replace_cuts(source.get_cut_list())
    assert len(cut_coeffs(replica)) == 1

    # simulate a purge on the source: the next snapshot no longer contains
    # the old cut, only a new one — replica must end up with just the new
    # one, not both.
    new_snapshot = [
        CutList([OptimalityCut(coeffs={0: 5.0}, rhs=9.0, info={}, objective_value=9.0)])
    ]
    replica.replace_cuts(new_snapshot)

    assert cut_coeffs(replica) == [{0: 5.0}]


def test_replace_cuts_is_a_noop_for_an_empty_snapshot():
    source = make_bm(purgeable=True)
    replica = make_bm(purgeable=False)

    source.add_cuts(
        [CutList([OptimalityCut(coeffs={0: 1.0}, rhs=2.0, info={}, objective_value=2.0)])]
    )
    replica.replace_cuts(source.get_cut_list())

    replica.replace_cuts([CutList()])

    assert cut_coeffs(replica) == []
