import pyomo.environ as pyo
from pyodec.core.dec_block import DecBlock
from pyodec.dec.ef.construct import ExtendedForm

# Create a model
model = pyo.ConcreteModel()

# Sets
CROPS = pyo.Set(initialize=['WHEAT', 'CORN', 'BEETS'])
SCENARIOS = pyo.Set(initialize=['GOOD', 'AVERAGE', 'POOR'])

# First stage parameters
model.TOTAL_ACREAGE = pyo.Param(initialize=500)
model.PlantingCostPerAcre = pyo.Param(
    CROPS, initialize={'WHEAT': 150.0, 'CORN': 230.0, 'BEETS': 260.0})

# First stage variables
model.DevotedAcreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)

# First stage objective
def first_stage_objective_rule(model):
    return -sum(model.PlantingCostPerAcre[crop] * model.DevotedAcreage[crop] for crop in CROPS)
model.objective = pyo.Objective(rule=first_stage_objective_rule, sense=pyo.maximize)

# First stage constraints
def land_constraint_rule(model):
    return sum(model.DevotedAcreage[crop] for crop in CROPS) <= model.TOTAL_ACREAGE
model.land_constraint = pyo.Constraint(rule=land_constraint_rule)

# Second stage

YIELD = {}
YIELD['GOOD'] = pyo.Param(
    CROPS, initialize={'WHEAT': 3.0, 'CORN': 3.6, 'BEETS': 24.0})
YIELD['AVERAGE'] = pyo.Param(
    CROPS, initialize={'WHEAT': 2.5, 'CORN': 3.0, 'BEETS': 20.0})
YIELD['POOR'] = pyo.Param(
    CROPS, initialize={'WHEAT': 2.0, 'CORN': 2.4, 'BEETS': 16.0})

def second_stage_rule(block, scenario):
    # Parameters
    block.Yield = YIELD[scenario]
    block.PriceQuota = pyo.Param(
        CROPS, initialize={'WHEAT': 100000.0, 'CORN': 100000.0, 'BEETS': 6000.0})
    block.SubQuotaSellingPrice = pyo.Param(
        CROPS, initialize={'WHEAT': 170.0, 'CORN': 150.0, 'BEETS': 36.0})
    block.SuperQuotaSellingPrice = pyo.Param(
        CROPS, initialize={'WHEAT': 0.0, 'CORN': 0.0, 'BEETS': 10.0})
    block.CattleFeedRequirement = pyo.Param(
        CROPS, initialize={'WHEAT': 200.0, 'CORN': 240.0, 'BEETS': 0.0})
    block.PurchasePrice = pyo.Param(
        CROPS, initialize={'WHEAT': 238.0, 'CORN': 210.0, 'BEETS': 100000.0})

    # Variables
    block.QuantitySubQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantitySuperQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantityRemainder = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
    block.QuantityPurchased = pyo.Var(CROPS, domain=pyo.NonNegativeReals)

    # Objective
    def profit_rule(block):
        return sum(block.SubQuotaSellingPrice[crop] * block.QuantitySubQuotaSold[crop] for crop in CROPS) + \
            sum(block.SuperQuotaSellingPrice[crop] * block.QuantitySuperQuotaSold[crop] for crop in CROPS) - \
            sum(block.PurchasePrice[crop] * block.QuantityPurchased[crop] for crop in CROPS)
    
    block.set_objective(pyo.Expression(rule=profit_rule))
    block.set_multiplier(1 / len(SCENARIOS))


    # Constraints

    def crop_selling_rule(block, crop):
        return block.QuantitySubQuotaSold[crop] + block.QuantitySuperQuotaSold[crop] + \
            block.QuantityRemainder[crop] == block.Yield[crop] * model.DevotedAcreage[crop]
    block.crop_selling_constraint = pyo.Constraint(CROPS, rule=crop_selling_rule)

    def cattle_feed_rule(block, crop):
        return block.QuantityRemainder[crop] + block.QuantityPurchased[crop] >= block.CattleFeedRequirement[crop]
    block.cattle_feed_constraint = pyo.Constraint(CROPS, rule=cattle_feed_rule)

    def quota_rule(block, crop):
        return block.QuantitySubQuotaSold[crop] <= block.PriceQuota[crop]
    block.quota_constraint = pyo.Constraint(CROPS, rule=quota_rule)

model.second_stage = DecBlock(SCENARIOS, rule=second_stage_rule)

# Solve
model = ExtendedForm(model).reconstruct()
solver = pyo.SolverFactory('appsi_highs')
solver.solve(model, tee=True)

# Display results
model.display()