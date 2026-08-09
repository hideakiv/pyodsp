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
from pyodsp.alg.const import (
    STATUS_NOT_FINISHED,
    STATUS_OPTIMAL,
    STATUS_MAX_ITERATION,
)
from pyodsp.dec.graph.tree import Tree
from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild, DecNodeInner


def make_leaf(idx):
    alg_leaf = FakeAlgLeaf()
    alg_leaf.get_init_up_message = MagicMock(return_value=FakeInitUpMessage())
    cut = OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)
    alg_leaf.get_up_message = MagicMock(return_value=FakeUpMessage(cut=cut))
    alg_leaf.get_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=1.0)
    )
    return DecNodeChild(idx=idx, alg_leaf=alg_leaf), alg_leaf


def make_root(idx=0):
    alg_root = FakeAlgRoot()
    alg_root.get_init_dn_message = MagicMock(
        return_value=FakeInitDnMessage(is_minimize=True)
    )
    alg_root.get_final_dn_message = MagicMock(return_value=FakeFinalDnMessage())
    alg_root.pass_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=42.0)
    )
    return DecNodeParent(idx=idx, alg_root=alg_root), alg_root


def test_verify_nodes_rejects_multiple_roots():
    root1, _ = make_root(0)
    root2, _ = make_root(1)
    leaf, _ = make_leaf(2)

    with pytest.raises(ValueError, match="Multiple root nodes"):
        Tree([root1, root2, leaf], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_rejects_unknown_node_type():
    # must at least look like an INode (Tree indexes by get_idx() before
    # checking type) but not satisfy INodeRoot/INodeLeaf/INodeInner.
    not_a_node = MagicMock(spec=[])
    not_a_node.get_idx = MagicMock(return_value=99)

    with pytest.raises(ValueError, match="Unknown object"):
        Tree([not_a_node], FakeLogger(), filedir=None, max_iteration=1)


def test_verify_nodes_accepts_inner_nodes():
    inner = DecNodeInner(idx=0, alg_root=FakeAlgRoot(), alg_leaf=FakeAlgLeaf())
    leaf, _ = make_leaf(1)

    tree = Tree([inner, leaf], FakeLogger(), filedir=None, max_iteration=1)

    assert tree.root is None  # inner nodes are skipped by root/leaf detection


def test_single_level_run_drives_root_and_leaf_to_completion(tmp_path):
    root, alg_root = make_root()
    leaf, alg_leaf = make_leaf(1)
    root.add_child(1)

    dn = FakeDnMessage(objective=10.0)
    alg_root.run_step = MagicMock(
        side_effect=[(STATUS_NOT_FINISHED, dn), (STATUS_OPTIMAL, dn)]
    )

    tree = Tree([root, leaf], FakeLogger(), filedir=tmp_path, max_iteration=10)
    tree.run()

    assert alg_root.run_step.call_count == 2
    alg_root.pass_final_up_message.assert_called_once()


def test_inner_node_max_iteration_still_adds_cuts_before_returning(tmp_path):
    top_root, top_alg_root = make_root(0)

    inner_alg_root = FakeAlgRoot()
    inner_alg_leaf = FakeAlgLeaf()
    inner_alg_leaf.get_init_up_message = MagicMock(return_value=FakeInitUpMessage())
    inner_alg_root.get_init_dn_message = MagicMock(
        return_value=FakeInitDnMessage(is_minimize=True)
    )
    inner = DecNodeInner(idx=1, alg_root=inner_alg_root, alg_leaf=inner_alg_leaf)

    leaf, alg_leaf = make_leaf(2)

    top_root.add_child(1)
    inner.add_child(2)

    dn_inner = FakeDnMessage(objective=5.0)
    inner_up_cut = OptimalityCut(coeffs={0: 1.0}, rhs=0.0, info={}, objective_value=0.0)
    inner_alg_root.run_step = MagicMock(
        return_value=(STATUS_MAX_ITERATION, dn_inner)
    )
    # a maxed-out inner node is asked for its up message via the *leaf* role
    # (DecNodeInner.get_up_message resolves to DecNodeChild.get_up_message).
    inner_alg_leaf.get_up_message = MagicMock(
        return_value=FakeUpMessage(cut=inner_up_cut)
    )
    inner_alg_root.add_cuts = MagicMock()

    top_dn = FakeDnMessage(objective=1.0)
    top_alg_root.run_step = MagicMock(
        side_effect=[(STATUS_NOT_FINISHED, top_dn), (STATUS_OPTIMAL, top_dn)]
    )
    top_alg_root.pass_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=99.0)
    )
    inner_alg_root.get_final_dn_message = MagicMock(return_value=FakeFinalDnMessage())
    inner_alg_root.pass_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=7.0)
    )
    inner_alg_leaf.get_final_up_message = MagicMock(
        return_value=FakeFinalUpMessage(objective=1.0)
    )

    tree = Tree([top_root, inner, leaf], FakeLogger(), filedir=tmp_path, max_iteration=10)
    tree.run()

    # inner hit STATUS_MAX_ITERATION -> the tree must call add_cuts once more
    # before treating it like a leaf and asking for its up message.
    inner_alg_root.add_cuts.assert_called_once()
    inner_alg_leaf.get_up_message.assert_called_once()
