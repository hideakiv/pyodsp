import pyomo.environ as pyo

def stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand, sc=0.5):
    block.produce = pyo.Var(domain=pyo.NonNegativeReals)
    block.overwork = pyo.Var(domain=pyo.NonNegativeReals)
    block.next_inventory = pyo.Var(domain=pyo.NonNegativeReals)

    block.c0 = pyo.Constraint(expr=block.produce <= 2)
    block.c1 = pyo.Constraint(expr=prev_inventory + block.produce + block.overwork - block.next_inventory == demand)

    return block.produce + 3.0 * block.overwork + sc * block.next_inventory

def first_stage(block: pyo.ConcreteModel, demand):
    block.prev_inventory = pyo.Var(domain=pyo.NonNegativeReals)
    block.prev_inventory.fix(0.0)
    objexpr = stage(block, block.prev_inventory, demand)
    block.obj = pyo.Objective(expr=objexpr, sense=pyo.minimize)

def mid_stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand):
    objexpr = stage(block, prev_inventory, demand)
    block.obj_expr = objexpr

def last_stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand):
    objexpr = stage(block, prev_inventory, demand, 0.0)
    block.obj_expr = objexpr