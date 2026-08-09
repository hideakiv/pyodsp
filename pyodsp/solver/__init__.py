"""The Pyomo wrapper every node holds.

Also where the objective sense is normalized: the algorithms only accept
minimization, so PyomoSolver converts a maximize model on construction and
remembers that it did, which is what lets results be reported in the user's
own units without anything downstream tracking a sign.
"""
