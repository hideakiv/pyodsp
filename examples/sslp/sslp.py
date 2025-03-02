"""
Source:
  Ntaimo, L. and S. Sen, "The 'million-variable' march for stochastic combinatorial optimization," Journal of Optimization, 2005.

Input:
  nJ: number of servers
  nI: number of clients
  nS: number of scenarios

Sets:
  sI: clients
  sJ: servers
  sZ: zones

Variables (1st Stage):
  x[j]: 1 if a server is located at site j, 0 otherwise

Variables (2nd Stage):
  y[i,j]: 1 if client i is served by a server at location j, 0 otherwise
  y0[j]: any overflows that are not served due to limitations in server capacity

Parameters (general):
  c[j]: cost of locating a server at location j
  q[i,j]: revenue from client i being served by server at location j
  q0[j]: overflow penalty
  d[i,j]: client i resource demand from server at location j
  u: server capacity
  v: an upper bound on the total number of servers that can be located
  w[z]: minimum number of servers to be located in zone z
  Jz[z]: the subset of server locations that belong to zone z
  p[s]: probability of occurrence for scenario s

Parameters (scenario):
  h[i,s]: 1 if client i is present in scenario s, 0 otherwise
"""
import random
import pyomo.environ as pyo
from pyomo.core.base.block import BlockData



def first_stage(model: pyo.ConcreteModel, nJ: int, seed: int=1):
    random.seed(seed)
    model.sJ = pyo.RangeSet(nJ)
    model.sZ = pyo.RangeSet(1) # for simplicity
    Jz = {1: model.sJ} # for simplicity

    def c_init(model, j):
        return random.randint(40, 80)
    model.c = pyo.Param(model.sJ, initialize=c_init)
    model.w = pyo.Param(model.sZ, default=0) # for simplicity

    v = nJ # for simplicity

    model.x = pyo.Var(model.sJ, domain=pyo.Binary)

    model.x_max = pyo.Constraint(expr=sum(model.x[j] for j in model.sJ) <= v)

    def rule_xmin(model, z):
        return sum(model.x[j] for j in Jz[z]) >= model.w[z]
    model.x_min = pyo.Constraint(model.sZ, rule=rule_xmin)

    model.obj = pyo.Objective(expr=sum(model.c[j] * model.x[j] for j in model.sJ), sense=pyo.minimize)\
    
def second_stage(block: BlockData, x: pyo.Var, nI: int, nJ: int, seed1: int=2, seed2: int=3):
    random.seed(seed1)
    block.sI = pyo.RangeSet(nI)
    block.sJ = pyo.RangeSet(nJ)

    def q_init(model, i, j):
        return random.randint(0, 25)
    block.q = pyo.Param(block.sI, block.sJ, initialize=q_init)
    block.q0 = pyo.Param(block.sJ, default=1000)

    def d_init(model, i, j): # for simplicity
        return model.q[i, j]
    block.d = pyo.Param(block.sI, block.sJ, initialize=d_init)
    u = 1.5 * sum(block.d[i,j] for i in block.sI for j in block.sJ) / nJ

    block.y = pyo.Var(block.sI, block.sJ, domain=pyo.Binary)
    block.y0 = pyo.Var(block.sJ, domain=pyo.NonNegativeReals)

    def rule_demand(block, j):
        return sum(block.d[i,j] * block.y[i,j] for i in block.sI) - block.y0[j] <= u * x[j]

    block.demand = pyo.Constraint(block.sJ, rule=rule_demand)

    random.seed(seed2)
    def h_init(model, i):
        return random.randint(0, 1)
    block.h = pyo.Param(block.sI, initialize=h_init)

    def rule_avail(block, i):
        return sum(block.y[i,j] for j in block.sJ) == block.h[i]
    block.avail = pyo.Constraint(block.sI, rule=rule_avail)

    block.objexpr = sum(block.q0[j] * block.y0[j] for j in block.sJ) \
        - sum(block.q[i,j] * block.y[i,j] for i in block.sI for j in block.sJ)


