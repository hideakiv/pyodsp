from pyomo.core.base.block import BlockData, CustomBlock, ScalarCustomBlockMixin

class DecBlockData(BlockData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_objective = 0.0
        self._block_multiplier = 1.0

    def set_objective(self, expr):
        #FIXME: warning - Reassigning the non-component attribute _block_objective
        self._block_objective = expr

    def get_objective(self):
        return self._block_objective

    def set_multiplier(self, multiplier):
        self._block_multiplier = multiplier

    def get_multiplier(self):
        return self._block_multiplier

class DecBlock(CustomBlock):
    
    def __init__(self, *args, **kwargs):
        self._ComponentDataClass = DecBlockData
        self._default_ctype = None
        super().__init__(*args, **kwargs)

class IndexedDecBlock(DecBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

DecBlock._indexed_custom_block = IndexedDecBlock

class ScalarDecBlock(ScalarCustomBlockMixin, DecBlockData, DecBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

DecBlock._scalar_custom_block = ScalarDecBlock
