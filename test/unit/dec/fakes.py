"""Minimal test doubles satisfying the node/alg/message/logger ABCs.

These exist so graph orchestration (Tree, HubAndSpoke, Lattice) and node
wiring (DecNode*) can be unit tested without a real Pyomo model or solver.
Every abstract method has a trivial concrete body; tests override the
specific methods they care about with `instance.method = MagicMock(...)`.
"""

from pyodsp.dec.node._alg import IAlgRoot, IAlgLeaf
from pyodsp.dec.node._logger import ILogger
from pyodsp.dec.node._message import (
    InitDnMessage,
    InitUpMessage,
    FinalUpMessage,
    FinalDnMessage,
    UpMessage,
    DnMessage,
)


class FakeAlgRoot(IAlgRoot):
    def set_logger(self, idx, depth, level):
        pass

    def save(self, dir):
        pass

    def is_minimize(self):
        return True

    def build(self, groups, children_multipliers, children_bounds):
        pass

    def run_step(self, up_messages):
        raise NotImplementedError

    def get_init_dn_message(self, **kwargs):
        raise NotImplementedError

    def add_cuts(self, up_messages):
        pass

    def reset_iteration(self):
        pass

    def get_final_dn_message(self, **kwargs):
        raise NotImplementedError

    def pass_final_up_message(self, messages):
        raise NotImplementedError

    def get_num_vars(self):
        return 0


class FakeAlgLeaf(IAlgLeaf):
    def set_logger(self, idx, depth, level):
        pass

    def save(self, dir):
        pass

    def is_minimize(self):
        return True

    def build(self):
        pass

    def pass_init_dn_message(self, message):
        pass

    def get_init_up_message(self):
        raise NotImplementedError

    def pass_dn_message(self, message):
        pass

    def pass_final_dn_message(self, message):
        pass

    def get_final_up_message(self):
        raise NotImplementedError

    def get_up_message(self):
        raise NotImplementedError


class FakeLogger(ILogger):
    def __init__(self):
        self.sub_problem_calls = []

    def log_info(self, text):
        pass

    def log_debug(self, text):
        pass

    def log_initialization(self, **kwargs):
        pass

    def log_master_problem(self, iteration, objective_value, x):
        pass

    def log_sub_problem(self, idx, cut_type, coefficients, constant):
        self.sub_problem_calls.append((idx, cut_type, coefficients, constant))

    def log_finaliziation(self):
        pass

    def log_completion(self, objective_value):
        pass


class FakeInitUpMessage(InitUpMessage):
    def __init__(self, bound=None):
        self.bound = bound

    def set_bound(self, bound):
        self.bound = bound

    def get_bound(self):
        return self.bound


class FakeFinalUpMessage(FinalUpMessage):
    def __init__(self, objective=None):
        self.objective = objective

    def get_objective(self):
        return self.objective

    def set_objective(self, obj):
        self.objective = obj


class FakeUpMessage(UpMessage):
    def __init__(self, cut=None, objective=0.0):
        self.cut = cut
        self.objective = objective

    def get_cut(self):
        return self.cut

    def get_objective(self):
        return self.objective


class FakeDnMessage(DnMessage):
    def __init__(self, objective=0.0):
        self.objective = objective

    def get_objective(self):
        return self.objective


class FakeInitDnMessage(InitDnMessage):
    def __init__(self, is_minimize=True, depth=None):
        self.is_minimize = is_minimize
        self.depth = depth

    def get_is_minimize(self):
        return self.is_minimize

    def set_depth(self, depth):
        self.depth = depth

    def get_depth(self):
        return self.depth


class FakeFinalDnMessage(FinalDnMessage):
    def __init__(self, solution=None):
        self.solution = solution

    def get_solution(self):
        return self.solution
