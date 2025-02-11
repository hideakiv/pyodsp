from pathlib import Path

from pyodec.dec.dd.run import DdRun

from equality import create_master, create_sub

from utils import get_args, assert_approximately_equal


def main():
    args = get_args()

    master = create_master("ipopt", pbm=True)
    sub_1 = create_sub(1, args.solver)
    sub_2 = create_sub(2, args.solver)
    sub_3 = create_sub(3, args.solver)

    master.add_child(1)
    master.add_child(2)
    master.add_child(3)

    dd_run = DdRun([master, sub_1, sub_2, sub_3], Path("output/dd/equality_pbm"))
    dd_run.run()

    assert_approximately_equal(master.alg.pbm.obj_bound[-1], -21.5)


if __name__ == "__main__":
    main()
