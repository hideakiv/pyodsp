import pyomo.environ as pyo

"""
bp_t:    the buy price at time t,
sp_t:    the selling price at time t, 
loss_t:    the proportion of flowers which survive from time t to t + 1, 
pc_t:    the procurement costs for extra flowers from another retailer,
demand_t:   the demand at time t.
"""

def stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand, bp=0.85, sp=1.0, loss=0.5, pc=1.25):
    block.purchase = pyo.Var(domain=pyo.NonNegativeReals)
    block.inventory_plus = pyo.Var(domain=pyo.NonNegativeReals)
    block.inventory_minus = pyo.Var(domain=pyo.NonNegativeReals)
    block.next_inventory = pyo.Var(domain=pyo.NonNegativeReals)

    block.c1 = pyo.Constraint(expr=prev_inventory - demand == block.inventory_plus - block.inventory_minus)
    block.c2 = pyo.Constraint(expr=block.next_inventory == block.purchase + loss * block.inventory_plus)

    return sp * demand - bp * block.purchase - pc * block.inventory_minus

def first_stage(block: pyo.ConcreteModel, bp=0.85):
    block.init_purchase = pyo.Var(domain=pyo.NonNegativeReals)
    block.obj = pyo.Objective(expr=-bp * block.init_purchase, sense=pyo.maximize)

def mid_stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand, bp=0.85, sp=1.0, loss=0.5, pc=1.25):
    objexpr = stage(block, prev_inventory, demand, bp, sp, loss, pc)
    block.obj_expr = objexpr

def last_stage(block: pyo.ScalarBlock, prev_inventory: pyo.Var, demand, bp=0.85, sp=1.0, loss=0.5, pc=1.25):
    objexpr = stage(block, prev_inventory, demand, bp, sp, loss, pc)
    block.purchase.fix(0.0)
    block.obj_expr = objexpr + block.next_inventory
