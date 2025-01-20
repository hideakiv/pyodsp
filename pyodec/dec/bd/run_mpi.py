from typing import List, Dict
from mpi4py import MPI

from .run import BdRun
from .node import BdNode


class BdRunMpi(BdRun):
    def __init__(self, nodes: List[BdNode], node_rank_map: Dict[int, int]):
        super().__init__(nodes)
        self.node_rank_map = node_rank_map
        self._get_nodes_by_parent()
        self._get_subgroups()
        self._communicate_groups()

    def _get_nodes_by_parent(self):
        self.nodes_by_parent: Dict[int, List[BdNode]] = {}
        for node in self.nodes.values():
            if node.parent in self.nodes_by_parent:
                self.nodes_by_parent[node.parent].append(node)
            else:
                self.nodes_by_parent[node.parent] = [node]
        if len(self.nodes_by_parent) > 1:
            raise NotImplementedError(
                "Currently only supports single parent per process"
            )

    def _get_subgroups(self):
        self.comm = MPI.COMM_WORLD
        self.subset_ranks: Dict[int, List[int]] = {}
        for idx, node in self.nodes.items():
            if len(node.children) > 0:
                ranks = self._get_subset_ranks(node)
                self.subset_ranks[idx] = ranks

    def _get_subset_ranks(self, node: BdNode) -> List[int]:
        ranks = set()
        ranks.add(self.node_rank_map[node.idx])
        for child in node.children:
            ranks.add(self.node_rank_map[child])
        ranks_list = list(ranks)
        ranks_list.insert(
            0, ranks_list.pop(ranks_list.index(self.node_rank_map[node.idx]))
        )
        return ranks_list

    def _create_comm_groups(self, group_members: List[int]):
        subgroup = self.comm.group.Incl(group_members)
        return self.comm.Create(subgroup)

    def _communicate_groups(self):
        for parent, node_list in self.nodes_by_parent.items():
            if parent is not None:
                group_members = self.comm.recv(source=self.node_rank_map[parent])

    def run(self):
        pass
