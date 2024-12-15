from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.PyomoModel import Model
from pyomo.core.base.var import VarData
from pyomo.core.base.block import BlockData
from pyodec.core.dec_block import DecBlockData

def get_vars_in_expression(expr, check_var) -> list:
    #TODO: maybe not sufficient for all cases
    vars = []
    if isinstance(expr, ExpressionBase):
        for i in range(expr.nargs()):
            vars.extend(get_vars_in_expression(expr.arg(i), check_var))
    elif isinstance(expr, VarData):
        if check_var(expr):
            vars.append(expr)
    return vars

def get_dec_block(comp) -> DecBlockData | Model:
    parent = comp.parent_block()
    if isinstance(parent, DecBlockData):
        return parent
    if isinstance(parent, Model):
        return parent
    return get_dec_block(parent)