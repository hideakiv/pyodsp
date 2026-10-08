from ..bdsc.alg_leaf_pyomo import BdScAlgLeafPyomo as SddpScAlgLeafPyomo

__all__ = ["SddpScAlgLeafPyomo"]

# A stage's leaf is exactly bdsc's two-stage leaf: its column-generation
# subproblem (CGSP) already embeds a free copy of whatever its model
# declares outside its own coupling variables (see bdsc's
# _verify_state_is_complete and examples/bdsc/cs.py, where the leaf embeds
# the root's first-stage model wholesale). For a stage-t leaf here, that
# model is the deterministic transition chain from stage t-K through t-1
# (K the lookback, see pyodsp.dec.sddpsc's module docstring), built as
# ordinary free, non-coupling variables -- only x_{t-1} is declared as the
# coupling variable, which is what keeps the resulting scaled cut a
# function of x_{t-1} alone despite the richer embedded feasible region.
# None of that is specific to this class; it's a property of which
# variables the model-building code marks as coupling, so there is nothing
# for sddpsc to override here.
