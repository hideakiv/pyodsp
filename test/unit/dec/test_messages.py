from pyodsp.alg.bm.cuts import OptimalityCut, CutList

from pyodsp.dec.bd.message import (
    BdInitDnMessage,
    BdInitUpMessage,
    BdUpMessage,
    BdDnMessage,
    BdFinalDnMessage,
    BdFinalUpMessage,
)
from pyodsp.dec.dd.message import (
    DdInitDnMessage,
    DdInitUpMessage,
    DdUpMessage,
    DdDnMessage,
    DdFinalDnMessage,
    DdFinalUpMessage,
)
from pyodsp.dec.bdsc.message import (
    BdScInitDnMessage,
    BdScInitUpMessage,
    BdScUpMessage,
    BdScDnMessage,
    BdScFinalDnMessage,
    BdScFinalUpMessage,
)


def make_cut():
    return OptimalityCut(coeffs={0: 1.0}, rhs=1.0, info={}, objective_value=1.0)


def test_bd_init_dn_message_roundtrip():
    message = BdInitDnMessage(is_minimize=True)
    message.set_depth(2)

    assert message.get_is_minimize() is True
    assert message.get_depth() == 2


def test_bd_init_up_message_defaults_to_no_bound():
    message = BdInitUpMessage()
    assert message.get_bound() is None
    message.set_bound(1.5)
    assert message.get_bound() == 1.5


def test_bd_up_message_holds_cut_and_objective():
    cut = make_cut()
    message = BdUpMessage(cut=cut, objective=3.0)
    assert message.get_cut() is cut
    assert message.get_objective() == 3.0


def test_bd_dn_message_defaults_objective_to_zero():
    message = BdDnMessage(solution=[1.0, 2.0])
    assert message.get_solution() == [1.0, 2.0]
    assert message.get_objective() == 0.0


def test_bd_final_messages_roundtrip():
    dn = BdFinalDnMessage(solution=[1.0])
    assert dn.get_solution() == [1.0]

    up = BdFinalUpMessage(objective=None)
    assert up.get_objective() is None
    up.set_objective(4.0)
    assert up.get_objective() == 4.0


def test_dd_init_dn_message_holds_coupling_matrix():
    matrix = [{0: 1.0}]
    message = DdInitDnMessage(coupling_matrix=matrix, is_minimize=False)
    message.set_depth(1)

    assert message.get_coupling_matrix() is matrix
    assert message.get_is_minimize() is False
    assert message.get_depth() == 1


def test_dd_up_message_objective_is_always_zero():
    message = DdUpMessage(cut=make_cut())
    assert message.get_objective() == 0.0


def test_dd_dn_message_objective_is_always_zero():
    message = DdDnMessage(solution=[1.0])
    assert message.get_solution() == [1.0]
    assert message.get_objective() == 0.0


def test_dd_final_up_message_carries_optional_solution():
    message = DdFinalUpMessage(objective=5.0, solution=[1.0, 2.0])
    assert message.get_objective() == 5.0
    assert message.get_solution() == [1.0, 2.0]
    message.set_objective(10.0)
    assert message.get_objective() == 10.0


def test_dd_final_dn_message_roundtrip():
    message = DdFinalDnMessage(solution=None)
    assert message.get_solution() is None


def test_bdsc_up_message_holds_c_and_tau():
    cut = make_cut()
    message = BdScUpMessage(cut=cut, c=0.1, tau=0.2, objective=1.0)

    assert message.get_cut() is cut
    assert message.get_c() == 0.1
    assert message.get_tau() == 0.2
    assert message.get_objective() == 1.0


def test_bdsc_dn_message_roundtrip():
    cut_list = [CutList([make_cut()])]
    message = BdScDnMessage(
        solution=[1.0], rho=0.5, cut_list=cut_list, subobj_bounds=[0.0]
    )

    assert message.get_solution() == [1.0]
    assert message.get_rho() == 0.5
    assert message.get_cut_list() is cut_list
    assert message.get_subobj_bounds() == [0.0]
    assert message.get_objective() == 0.0


def test_bdsc_init_and_final_messages_roundtrip():
    init_dn = BdScInitDnMessage(is_minimize=True)
    init_dn.set_depth(3)
    assert init_dn.get_is_minimize() is True
    assert init_dn.get_depth() == 3

    init_up = BdScInitUpMessage()
    assert init_up.get_bound() is None

    final_dn = BdScFinalDnMessage(solution=[1.0])
    assert final_dn.get_solution() == [1.0]

    final_up = BdScFinalUpMessage(objective=1.0)
    final_up.set_objective(2.0)
    assert final_up.get_objective() == 2.0
