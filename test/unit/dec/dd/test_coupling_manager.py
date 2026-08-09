"""The coupling manager works purely in the minimize convention.

It used to branch on the problem's sense; PyomoSolver now converts a
maximize model on construction, so there is only one convention left
here and the sign choices below are unconditional.
"""

from pyodsp.dec.dd.coupling_manager import CouplingManager


def test_convert_to_col_major():
    # row 0: var0=1, var1=2 ; row 1: var1=3
    matrix = [{0: 1.0, 1: 2.0}, {1: 3.0}]
    manager = CouplingManager(matrix, len_vars=2)

    assert manager.col_major == [{0: 1.0}, {0: 2.0, 1: 3.0}]


def test_dual_times_matrix_keeps_sign():
    matrix = [{0: 1.0, 1: 2.0}]
    manager = CouplingManager(matrix, len_vars=2)

    result = manager.dual_times_matrix([3.0])

    assert result == [3.0, 6.0]


def test_matrix_times_primal_flips_sign():
    matrix = [{0: 1.0, 1: 2.0}]
    manager = CouplingManager(matrix, len_vars=2)

    result = manager.matrix_times_primal([2.0, 5.0])

    # row: 1*2 + 2*5 = 12, negated for the minimize convention
    assert result == [-12.0]


def test_inner_product():
    manager = CouplingManager([], len_vars=0)

    assert manager.inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0


def test_inner_product_with_unequal_length_stops_at_shortest():
    manager = CouplingManager([], len_vars=0)

    assert manager.inner_product([1.0, 2.0], [4.0, 5.0, 6.0]) == 14.0
