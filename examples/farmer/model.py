import pyomo.environ as pyo

# Create a model
model = pyo.ConcreteModel()

# Sets
CROPS = pyo.Set(initialize=["WHEAT", "CORN", "BEETS"])

# Parameters
model.Yield = pyo.Param(CROPS, initialize={"WHEAT": 2.5, "CORN": 3, "BEETS": 20})

model.TOTAL_ACREAGE = pyo.Param(initialize=500)
model.PriceQuota = pyo.Param(
    CROPS, initialize={"WHEAT": 100000.0, "CORN": 100000.0, "BEETS": 6000.0}
)
model.SubQuotaSellingPrice = pyo.Param(
    CROPS, initialize={"WHEAT": 170.0, "CORN": 150.0, "BEETS": 36.0}
)
model.SuperQuotaSellingPrice = pyo.Param(
    CROPS, initialize={"WHEAT": 0.0, "CORN": 0.0, "BEETS": 10.0}
)
model.CattleFeedRequirement = pyo.Param(
    CROPS, initialize={"WHEAT": 200.0, "CORN": 240.0, "BEETS": 0.0}
)
model.PurchasePrice = pyo.Param(
    CROPS, initialize={"WHEAT": 238.0, "CORN": 210.0, "BEETS": 100000.0}
)
model.PlantingCostPerAcre = pyo.Param(
    CROPS, initialize={"WHEAT": 150.0, "CORN": 230.0, "BEETS": 260.0}
)

# Variables
model.DevotedAcreage = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
model.QuantitySubQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
model.QuantitySuperQuotaSold = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
model.QuantityRemainder = pyo.Var(CROPS, domain=pyo.NonNegativeReals)
model.QuantityPurchased = pyo.Var(CROPS, domain=pyo.NonNegativeReals)


# Objective
def profit_rule(model):
    return (
        sum(
            model.SubQuotaSellingPrice[crop] * model.QuantitySubQuotaSold[crop]
            for crop in CROPS
        )
        + sum(
            model.SuperQuotaSellingPrice[crop] * model.QuantitySuperQuotaSold[crop]
            for crop in CROPS
        )
        - sum(
            model.PurchasePrice[crop] * model.QuantityPurchased[crop] for crop in CROPS
        )
        - sum(
            model.PlantingCostPerAcre[crop] * model.DevotedAcreage[crop]
            for crop in CROPS
        )
    )


model.profit = pyo.Objective(rule=profit_rule, sense=pyo.maximize)


# Constraints
def land_constraint_rule(model):
    return sum(model.DevotedAcreage[crop] for crop in CROPS) <= model.TOTAL_ACREAGE


model.land_constraint = pyo.Constraint(rule=land_constraint_rule)


def crop_selling_rule(model, crop):
    return (
        model.QuantitySubQuotaSold[crop]
        + model.QuantitySuperQuotaSold[crop]
        + model.QuantityRemainder[crop]
        == model.Yield[crop] * model.DevotedAcreage[crop]
    )


model.crop_selling_constraint = pyo.Constraint(CROPS, rule=crop_selling_rule)


def cattle_feed_rule(model, crop):
    return (
        model.QuantityRemainder[crop] + model.QuantityPurchased[crop]
        >= model.CattleFeedRequirement[crop]
    )


model.cattle_feed_constraint = pyo.Constraint(CROPS, rule=cattle_feed_rule)


def quota_rule(model, crop):
    return model.QuantitySubQuotaSold[crop] <= model.PriceQuota[crop]


model.quota_constraint = pyo.Constraint(CROPS, rule=quota_rule)

# Solve
solver = pyo.SolverFactory("appsi_highs")
solver.solve(model, tee=True)

# Display results
model.display()
