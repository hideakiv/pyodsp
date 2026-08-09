"""Birge and Louveaux's farmer, wired to pyodsp.dec by hand.

The farmer maximizes profit, but the decomposition algorithms only accept
minimize problems, so this runs the three-step conversion documented on
pyomo_utils.negate_objective_sense: negate every node's model before
building its PyomoSolver, negate the bound passed to set_bound (it is in
the same true-objective units), and negate the saved trajectory once the
run is over. Values read back off a node are then in the internal
(negated) convention and are negated to report.

See sp_pipeline.py for the same problem through pyodsp.model.sp, which
does all of that — and the coupling-variable bookkeeping — itself.
"""

from pathlib import Path

import pyomo.environ as pyo

from pyodsp.solver.pyomo_solver import PyomoSolver, SolverConfig
from pyodsp.solver.pyomo_utils import (
    negate_objective_sense,
    negate_saved_objective_csv,
)

from pyodsp.dec.node.dec_node import DecNodeRoot, DecNodeLeaf
from pyodsp.dec.bd.alg_root_bm import BdAlgRootBm
from pyodsp.dec.bd.alg_leaf_pyomo import BdAlgLeafPyomo
from pyodsp.dec.bd.run import BdRun

FILEDIR = Path("output/farmer/bd_model")
OPTIMAL_PROFIT = 108390.0
# An upper bound on any one scenario's recourse profit, in true units.
RECOURSE_PROFIT_BOUND = 1000000.0

# Create a model
model = pyo.ConcreteModel()

# Sets
CROPS = pyo.Set(initialize=["WHEAT", "CORN", "BEETS"])
SCENARIOS = ["GOOD", "AVERAGE", "POOR"]

# First stage parameters
model.TOTAL_ACREAGE = pyo.Param(initialize=500)
model.PlantingCostPerAcre = pyo.Param(
    CROPS, initialize={"WHEAT": 150.0, "CORN": 230.0, "BEETS": 260.0}
)

# First stage variables
model.DevotedAcreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)


# First stage constraints
def land_constraint_rule(model):
    return sum(model.DevotedAcreage[crop] for crop in CROPS) <= model.TOTAL_ACREAGE


model.land_constraint = pyo.Constraint(rule=land_constraint_rule)


# First stage objective
def objective_rule(model):
    return -sum(
        model.PlantingCostPerAcre[crop] * model.DevotedAcreage[crop] for crop in CROPS
    )


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

# Convert to the minimize form the algorithms require. This has to happen
# before the PyomoSolver is built: the solver captures the active
# objective as the one it reports as "original".
negate_objective_sense(model)

coupling_dn = [model.DevotedAcreage[crop] for crop in CROPS]
config = SolverConfig(solver_name="appsi_highs")
first_stage_solver = PyomoSolver(model, config, coupling_dn)
first_stage_alg = BdAlgRootBm(first_stage_solver)
root_node = DecNodeRoot(0, first_stage_alg)


# Second stage

second_stage = {scenario: pyo.ConcreteModel() for scenario in SCENARIOS}

YIELD = {}
YIELD["GOOD"] = pyo.Param(CROPS, initialize={"WHEAT": 3.0, "CORN": 3.6, "BEETS": 24.0})
YIELD["AVERAGE"] = pyo.Param(
    CROPS, initialize={"WHEAT": 2.5, "CORN": 3.0, "BEETS": 20.0}
)
YIELD["POOR"] = pyo.Param(CROPS, initialize={"WHEAT": 2.0, "CORN": 2.4, "BEETS": 16.0})

for scenario, block in second_stage.items():
    # Parameters
    block.Yield = YIELD[scenario]
    block.PriceQuota = pyo.Param(
        CROPS, initialize={"WHEAT": 100000.0, "CORN": 100000.0, "BEETS": 6000.0}
    )
    block.SubQuotaSellingPrice = pyo.Param(
        CROPS, initialize={"WHEAT": 170.0, "CORN": 150.0, "BEETS": 36.0}
    )
    block.SuperQuotaSellingPrice = pyo.Param(
        CROPS, initialize={"WHEAT": 0.0, "CORN": 0.0, "BEETS": 10.0}
    )
    block.CattleFeedRequirement = pyo.Param(
        CROPS, initialize={"WHEAT": 200.0, "CORN": 240.0, "BEETS": 0.0}
    )
    block.PurchasePrice = pyo.Param(
        CROPS, initialize={"WHEAT": 238.0, "CORN": 210.0, "BEETS": 100000.0}
    )

    # Variables
    block.QuantitySubQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantitySuperQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantityRemainder = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantityPurchased = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.DevotedAcreage = pyo.Var(CROPS, domain=pyo.Reals)

    # Constraints

    def crop_selling_rule(block, crop):
        return (
            block.QuantitySubQuotaSold[crop]
            + block.QuantitySuperQuotaSold[crop]
            + block.QuantityRemainder[crop]
            == block.Yield[crop] * block.DevotedAcreage[crop]
        )

    block.crop_selling_constraint = pyo.Constraint(CROPS, rule=crop_selling_rule)

    def cattle_feed_rule(block, crop):
        return (
            block.QuantityRemainder[crop] + block.QuantityPurchased[crop]
            >= block.CattleFeedRequirement[crop]
        )

    block.cattle_feed_constraint = pyo.Constraint(CROPS, rule=cattle_feed_rule)

    def quota_rule(block, crop):
        return block.QuantitySubQuotaSold[crop] <= block.PriceQuota[crop]

    block.quota_constraint = pyo.Constraint(CROPS, rule=quota_rule)

    # second stage objective

    def profit_rule(block):
        profit = (
            sum(
                block.SubQuotaSellingPrice[crop] * block.QuantitySubQuotaSold[crop]
                for crop in CROPS
            )
            + sum(
                block.SuperQuotaSellingPrice[crop] * block.QuantitySuperQuotaSold[crop]
                for crop in CROPS
            )
            - sum(
                block.PurchasePrice[crop] * block.QuantityPurchased[crop]
                for crop in CROPS
            )
        )
        return profit

    block.objective = pyo.Objective(rule=profit_rule, sense=pyo.maximize)

second_stage_solver = {}
for scenario, block in second_stage.items():
    # Every node's model gets the same conversion, not just the root: the
    # leaf algorithm rejects a maximize model too, and the two senses have
    # to agree for the cuts to mean anything.
    negate_objective_sense(block)

    coupling_vars_up = [block.DevotedAcreage[crop] for crop in CROPS]
    config = SolverConfig(solver_name="appsi_highs")
    second_stage_solver[scenario] = PyomoSolver(block, config, coupling_vars_up)


leaf_nodes = {}
idx = 1
for scenario, block in second_stage.items():
    alg = BdAlgLeafPyomo(second_stage_solver[scenario])
    leaf_node = DecNodeLeaf(idx, alg)
    # The bound is in true objective units, so it negates with everything
    # else: an upper bound on profit is a lower bound on negated profit.
    leaf_node.set_bound(-RECOURSE_PROFIT_BOUND)
    leaf_nodes[scenario] = leaf_node
    root_node.add_child(idx, multiplier=1 / len(SCENARIOS))
    idx += 1

bd_run = BdRun([root_node, *leaf_nodes.values()], FILEDIR)
bd_run.run()

# The graph saves whatever sense the node's model had and has no notion of
# "this was negated", so the root's trajectory is left in internal units
# until it is corrected here. sol.csv needs no such fix, and the leaves
# save no trajectory.
negate_saved_objective_csv(FILEDIR / "node0")

# Values read off the nodes are still in the internal convention.
profit = -first_stage_alg.bm.obj_bound[-1]

print(f"expected profit: {profit:,.2f}")
for crop in CROPS:
    print(f"  {crop}: {model.DevotedAcreage[crop].value:,.2f} acres")

assert abs(profit - OPTIMAL_PROFIT) < 1e-3, f"expected {OPTIMAL_PROFIT}, got {profit}"
