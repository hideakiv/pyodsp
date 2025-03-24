
from ..node.dec_node import DecNodeRoot
from .alg_root import BdAlgRoot


class BdRootNode(DecNodeRoot):

    def __init__(
        self,
        idx: int,
        alg_root: BdAlgRoot,
    ) -> None:
        super().__init__(idx, alg_root)


