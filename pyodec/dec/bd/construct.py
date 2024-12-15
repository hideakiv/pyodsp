from pyomo.core.base.constraint import ConstraintData, IndexedConstraint
from pyomo.core.base.block import BlockData, IndexedBlock
from pyodec.core.dec_block import DecBlockData, IndexedDecBlock
from ..utils import get_vars_in_expression, get_dec_block

class BendersDec:
    def __init__(self, source):
        self.source = source

    def reconstruct(self):
        self._reconstruct_root(self.source)

    def _reconstruct_root(self, block: BlockData):
        for key, value in block.component_map().items():
            if isinstance(value, ConstraintData):
                if self._contains_other_dec_block_vars(value):
                    raise ValueError("Constraint refers to sub-dec_block variables")
            elif isinstance(value, IndexedConstraint):
                if self._indexed_contains_other_dec_block_vars(value):
                    raise ValueError("Constraint refers to sub-dec_block variables")
            elif isinstance(value, DecBlockData):
                self._reconstruct_dec_block_data(value)
            elif isinstance(value, IndexedDecBlock):
                self._reconstruct_indexed_dec_block(value)
            elif isinstance(value, BlockData):
                self._reconstruct_root(value)
            elif isinstance(value, IndexedBlock):
                self._indexed_reconstruct_root(value)
    
    def _indexed_reconstruct_root(self, indexed_block: IndexedBlock):
        for value in indexed_block.values():
            self._reconstruct_root(value)

    def _contains_other_dec_block_vars(self, constraint: ConstraintData) -> bool:
        """
        check if the constraint refers to other dec_block variables
        """
        vars = self._get_other_dec_block_vars(constraint)
        return len(vars) > 0
    
    def _get_other_dec_block_vars(self, constraint: ConstraintData) -> list:
        current_block = get_dec_block(constraint)

        def check_var(var):
            return get_dec_block(var) != current_block

        return get_vars_in_expression(constraint.expr, check_var)

    def _indexed_contains_other_dec_block_vars(self, indexed_constraint: IndexedConstraint) -> bool:
        for value in indexed_constraint.values():
            if self._contains_other_dec_block_vars(value):
                return True
        return False

    def _reconstruct_dec_block_data(self, dec_block_data: DecBlockData):
        pass

    def _reconstruct_indexed_dec_block(self, indexed_dec_block: IndexedDecBlock):
        for value in indexed_dec_block.values():
            self._reconstruct_dec_block_data(value)
    
    def _reconstruct_block(self, block: BlockData):
        for key, value in block.component_map().items():
            if isinstance(value, ConstraintData):
                self._check_constraint(value)
            elif isinstance(value, IndexedConstraint):
                self._check_indexed_constraint(value)
            elif isinstance(value, DecBlockData):
                self._reconstruct_dec_block_data(value)
            elif isinstance(value, IndexedDecBlock):
                self._reconstruct_indexed_dec_block(value)
            elif isinstance(value, BlockData):
                self._reconstruct_block(value)
            elif isinstance(value, IndexedBlock):
                self._reconstruct_indexed_block(value)
    
    def _reconstruct_indexed_block(self, indexed_block: IndexedBlock):
        for value in indexed_block.values():
            self._reconstruct_block(value)
        