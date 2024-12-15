from pyomo.environ import Objective
from pyomo.core.base.block import BlockData, IndexedBlock
from pyodec.core.dec_block import DecBlockData, IndexedDecBlock


class ExtendedForm:
    def __init__(self, source):
        self.source = source

    def reconstruct(self):
        self._objective_key = None
        expr = 0.0
        for key, value in self.source.component_map().items():
            if isinstance(value, Objective):
                if self._objective_key is not None:
                    raise ValueError("Multiple objectives found")
                self._objective_key = key
            elif isinstance(value, DecBlockData):
                expr += self._reconstruct_dec_block_data(value)
            elif isinstance(value, IndexedDecBlock):
                expr += self._reconstruct_indexed_dec_block(value)
            elif isinstance(value, BlockData):
                expr += self._reconstruct_block(value)
            elif isinstance(value, IndexedBlock):
                expr += self._reconstruct_indexed_block(value)
        
        self.source.__dict__[self._objective_key].expr += expr
        return self.source
    
    def _reconstruct_dec_block_data(self, dec_block_data: DecBlockData):
        expr = 0.0
        expr += self._reconstruct_block(dec_block_data)
        expr += dec_block_data._block_objective
        return dec_block_data._block_multiplier * expr

    def _reconstruct_indexed_dec_block(self, indexed_dec_block: IndexedDecBlock):
        expr = 0.0
        for value in indexed_dec_block.values():
            expr += self._reconstruct_dec_block_data(value)
        return expr

    def _reconstruct_block(self, block: BlockData | DecBlockData):
        expr = 0.0
        for key, value in block.component_map().items():
            if isinstance(value, Objective):
                raise ValueError("Objective found in BlockData")
            elif isinstance(value, DecBlockData):
                expr += self._reconstruct_dec_block_data(value)
            elif isinstance(value, IndexedDecBlock):
                expr += self._reconstruct_indexed_dec_block(value)
            elif isinstance(value, BlockData):
                expr += self._reconstruct_block(value)
            elif isinstance(value, IndexedBlock):
                expr += self._reconstruct_indexed_block(value)
        return expr
    
    def _reconstruct_indexed_block(self, indexed_block: IndexedBlock):
        expr = 0.0
        for value in indexed_block.values():
            expr += self._reconstruct_block(value)
        return expr


