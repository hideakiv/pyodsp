from pathlib import Path

from pyodec.dec.dd.run import DdRun

from .equality import create_master, create_sub


def main():

    master = create_master("ipopt", pbm=True)
    sub_1 = create_sub(1)
    sub_2 = create_sub(2)
    sub_3 = create_sub(3)

    master.add_child(1)
    master.add_child(2)
    master.add_child(3)

    dd_run = DdRun([master, sub_1, sub_2, sub_3], Path("output/dd/equality_pbm"))
    dd_run.run()


if __name__ == "__main__":
    main()
