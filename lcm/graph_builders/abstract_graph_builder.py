import networkx as nx
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Hashable, List, Tuple

Step = Tuple[Hashable, Hashable, str]


class AbstractGraphBuilder(ABC):
    """Abstract base class for all graph builder"""

    # Draw the root node even when there are no cases.
    ROOT_ALWAYS = False

    @abstractmethod
    def root(self) -> Hashable:
        """The state every case starts in."""

    @abstractmethod
    def walk(self, variant: List[str], allow_loops: bool) -> List[Step]:
        """Return the (from_state, to_state, activity) steps one variant takes.

        The states are the node keys of the graph `build` returns. A step that
        `build` does not draw (the state is unchanged and loops are hidden) is
        left out.
        """

    @abstractmethod
    def node_attributes(self, state: Hashable) -> Dict[str, Any]:
        """The attributes of the node of `state`, without its weight."""

    def build(
        self, variants: List[List[str]], allow_loops: bool = True
    ) -> nx.MultiDiGraph:
        """Draw every step of every variant into one graph."""
        root = self.root()
        state_counts = defaultdict(int)
        edge_counts = defaultdict(int)
        if self.ROOT_ALWAYS:
            state_counts[root] = 0

        for variant in variants:
            state_counts[root] += 1
            for from_state, to_state, activity in self.walk(variant, allow_loops):
                state_counts[to_state] += 1
                edge_counts[(from_state, to_state, activity)] += 1

        G = nx.MultiDiGraph()
        for state, count in state_counts.items():
            G.add_node(state, weight=count, **self.node_attributes(state))
        for (from_state, to_state, activity), count in edge_counts.items():
            G.add_edge(from_state, to_state, weight=count, activity=activity)
        return G

    @property
    @abstractmethod
    def name(self) -> str:
        """User friendly name for the builder strategy"""
