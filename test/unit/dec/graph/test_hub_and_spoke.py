from unittest.mock import MagicMock

import pytest

from fakes import (
    FakeAlgRoot,
    FakeAlgLeaf,
    FakeLogger,
    FakeInitDnMessage,
    FakeInitUpMessage,
    FakeUpMessage,
    FakeDnMessage,
    FakeFinalDnMessage,
    FakeFinalUpMessage,
)

from pyodsp.alg.bm.cuts import OptimalityCut
from pyodsp.alg.const import STATUS_NOT_FINISHED, STATUS_OPTIMAL
from pyodsp.dec.graph.hub_and_spoke import HubAndSpoke
from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild, DecNodeInner


def make_root_and_leaf():
    alg_root = FakeAlgRoot()
    alg_root.get_init_dn_message = MagicMock(
        return_value=FakeInitDnMessage(is_minimize=True)
    )
    alg_root.get_final_dn_message = MagicMock(return_value=FakeFinalDnMessage())
    alg_root.pass_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=42.0)
    )
    root = DecNodeParent(idx=0, alg_root=alg_root)

    alg_leaf = FakeAlgLeaf()
    alg_leaf.get_init_up_message = MagicMock(return_value=FakeInitUpMessage())
    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    alg_leaf.get_up_message = MagicMock(return_value=FakeUpMessage(cut=cut))
    alg_leaf.get_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=1.0)
    )
    leaf = DecNodeChild(idx=1, alg_leaf=alg_leaf)

    root.add_child(1)
    return root, leaf, alg_root, alg_leaf


def test_verify_nodes_rejects_multiple_roots():
    root1, leaf, _, _ = make_root_and_leaf()
    root2, _, _, _ = make_root_and_leaf()

    with pytest.raises(ValueError, match="Multiple root nodes"):
        HubAndSpoke(
            [root1, root2, leaf], FakeLogger(), filedir=None, max_iteration=1
        )


def test_verify_nodes_rejects_inner_nodes():
    inner = DecNodeInner(idx=0, alg_root=FakeAlgRoot(), alg_leaf=FakeAlgLeaf())

    with pytest.raises(ValueError, match="prohibited in HubAndSpoke"):
        HubAndSpoke([inner], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_rejects_unknown_node_type():
    with pytest.raises(ValueError, match="Unknown object"):
        HubAndSpoke([object()], FakeLogger(), filedir=None, max_iteration=1)


def test_run_drives_root_and_leaf_to_completion(tmp_path):
    root, leaf, alg_root, alg_leaf = make_root_and_leaf()

    dn1 = FakeDnMessage(objective=10.0)
    alg_root.run_step = MagicMock(
        side_effect=[(STATUS_NOT_FINISHED, dn1), (STATUS_OPTIMAL, dn1)]
    )

    graph = HubAndSpoke([root, leaf], FakeLogger(), filedir=tmp_path, max_iteration=10)
    graph.run()

    assert alg_root.run_step.call_count == 2
    # the leaf only solves for iterations where the loop continues
    # (STATUS_NOT_FINISHED); the final STATUS_OPTIMAL iteration breaks first.
    assert alg_leaf.get_up_message.call_count == 1
    alg_root.pass_final_up_message.assert_called_once()


def test_run_stops_at_max_iteration_if_never_optimal(tmp_path):
    root, leaf, alg_root, alg_leaf = make_root_and_leaf()

    dn = FakeDnMessage(objective=10.0)
    alg_root.run_step = MagicMock(return_value=(STATUS_NOT_FINISHED, dn))

    graph = HubAndSpoke([root, leaf], FakeLogger(), filedir=tmp_path, max_iteration=3)
    graph.run()

    assert alg_root.run_step.call_count == 3


def test_run_final_returns_root_objective(tmp_path):
    root, leaf, alg_root, alg_leaf = make_root_and_leaf()
    dn = FakeDnMessage(objective=10.0)
    alg_root.run_step = MagicMock(return_value=(STATUS_OPTIMAL, dn))

    graph = HubAndSpoke([root, leaf], FakeLogger(), filedir=tmp_path, max_iteration=10)
    graph.run()

    (call_args,) = alg_root.pass_final_up_message.call_args_list
    up_messages = call_args.args[0]
    assert up_messages[1].get_objective() == 1.0
