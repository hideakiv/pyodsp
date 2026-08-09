from unittest.mock import MagicMock

import pytest

from fakes import FakeAlgRoot, FakeAlgLeaf, FakeInitUpMessage, FakeFinalUpMessage

from pyodsp.dec.node.dec_node import DecNodeParent, DecNodeChild, DecNodeInner


def test_build_is_idempotent():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.build_inner = MagicMock()

    node.build()
    node.build()

    node.build_inner.assert_called_once()


def test_add_child_registers_multiplier():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())

    node.add_child(1, multiplier=2.5)

    assert node.get_children() == [1]
    assert node.get_multiplier(1) == 2.5


def test_add_child_raises_on_duplicate():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.add_child(1)

    with pytest.raises(ValueError):
        node.add_child(1)


def test_set_groups_raises_when_groups_overlap():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.add_child(1)
    node.add_child(2)

    with pytest.raises(ValueError, match="not disjoint"):
        node.set_groups([[1, 2], [2]])


def test_set_groups_raises_when_groups_do_not_cover_children():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.add_child(1)
    node.add_child(2)

    with pytest.raises(ValueError, match="Not all elements"):
        node.set_groups([[1]])


def test_set_groups_accepts_disjoint_covering_groups():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.add_child(1)
    node.add_child(2)

    node.set_groups([[1], [2]])

    assert node.get_groups() == [[1], [2]]


def test_build_inner_defaults_to_singleton_groups_when_none_set():
    alg_root = FakeAlgRoot()
    alg_root.build = MagicMock()
    node = DecNodeParent(idx=0, alg_root=alg_root)
    node.add_child(1)
    node.add_child(2)

    node.build_inner()

    assert node.groups == [[1], [2]]
    alg_root.build.assert_called_once_with(
        [[1], [2]], node.children_multipliers, node.children_bounds
    )


def test_pass_init_up_messages_skips_none_bound():
    node = DecNodeParent(idx=0, alg_root=FakeAlgRoot())
    node.add_child(1)
    node.add_child(2)

    node.pass_init_up_messages(
        {1: FakeInitUpMessage(bound=None), 2: FakeInitUpMessage(bound=5.0)}
    )

    assert 1 not in node.children_bounds
    assert node.get_child_bound(2) == 5.0


def test_pass_final_up_message_scales_objective_by_multiplier():
    alg_root = FakeAlgRoot()
    alg_root.pass_final_up_message = MagicMock(side_effect=lambda messages: messages)
    node = DecNodeParent(idx=0, alg_root=alg_root)
    node.add_child(1, multiplier=3.0)

    messages = {1: FakeFinalUpMessage(objective=2.0)}
    node.pass_final_up_message(messages)

    assert messages[1].get_objective() == 6.0


def test_pass_final_up_message_does_not_scale_none_objective():
    alg_root = FakeAlgRoot()
    alg_root.pass_final_up_message = MagicMock(side_effect=lambda messages: messages)
    node = DecNodeParent(idx=0, alg_root=alg_root)
    node.add_child(1, multiplier=3.0)

    messages = {1: FakeFinalUpMessage(objective=None)}
    node.pass_final_up_message(messages)

    assert messages[1].get_objective() is None


def test_child_pass_init_dn_message_sets_depth_from_parent_depth():
    alg_leaf = FakeAlgLeaf()
    alg_leaf.pass_init_dn_message = MagicMock()
    node = DecNodeChild(idx=1, alg_leaf=alg_leaf)
    init_message = MagicMock()
    init_message.get_depth.return_value = 4

    node.pass_init_dn_message(init_message)

    assert node.get_depth() == 5


def test_get_depth_raises_when_not_initialized():
    node = DecNodeChild(idx=1, alg_leaf=FakeAlgLeaf())

    with pytest.raises(ValueError):
        node.get_depth()


def test_dec_node_inner_build_inner_calls_both_parent_and_child():
    alg_root = FakeAlgRoot()
    alg_root.build = MagicMock()
    alg_leaf = FakeAlgLeaf()
    alg_leaf.build = MagicMock()
    node = DecNodeInner(idx=0, alg_root=alg_root, alg_leaf=alg_leaf)

    node.build_inner()

    alg_root.build.assert_called_once()
    alg_leaf.build.assert_called_once()
