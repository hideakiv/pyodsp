from pyomo.core.base.block import BlockData, CustomBlock, ScalarCustomBlockMixin

class DecBlockData(BlockData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_objective = 0.0
        self._block_multiplier = 1.0

    def set_objective(self, expr):
        self._block_objective = expr

    def set_multiplier(self, multiplier):
        self._block_multiplier = multiplier

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
