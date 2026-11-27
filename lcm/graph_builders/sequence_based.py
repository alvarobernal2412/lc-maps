from typing import Any, Dict, List, Tuple

from .abstract_graph_builder import AbstractGraphBuilder, Step


class SequenceBasedPrefixTreeBuilder(AbstractGraphBuilder):
    """
    Build a prefix tree where each node is a unique sequence (tuple) of activities.
    No merging: A->B->C is different from B->A->C.
    """

    @property
    def name(self) -> str:
        return "Sequence-Based Prefix Tree"

    def root(self) -> Tuple[str, ...]:
        return tuple()  # empty prefix

    def walk(self, variant: List[str], allow_loops: bool) -> List[Step]:
        # Every prefix is a new state, so no step is ever hidden.
        return [
            (tuple(variant[:i]), tuple(variant[: i + 1]), activity)
            for i, activity in enumerate(variant)
        ]

    def node_attributes(self, state: Tuple[str, ...]) -> Dict[str, Any]:
        # Label: START for empty, otherwise "A → B → C"
        return {
            "label": "START" if len(state) == 0 else " → ".join(state),
            "sequence_length": len(state),
            "activities": list(state),
        }
