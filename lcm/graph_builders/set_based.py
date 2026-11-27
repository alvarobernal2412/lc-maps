from typing import Any, Dict, FrozenSet, List

from .abstract_graph_builder import AbstractGraphBuilder, Step


class SetBasedPrefixGraphBuilder(AbstractGraphBuilder):
    """Build a prefix graph where each node is the set of activities done so far."""

    @property
    def name(self) -> str:
        return "Set-Based Prefix Graph"

    def root(self) -> FrozenSet[str]:
        return frozenset()

    def walk(self, variant: List[str], allow_loops: bool) -> List[Step]:
        steps = []
        current_state = frozenset()
        for i, activity in enumerate(variant):
            new_state = frozenset(variant[: i + 1])
            if current_state != new_state or allow_loops:
                steps.append((current_state, new_state, activity))
            current_state = new_state
        return steps

    def node_attributes(self, state: FrozenSet[str]) -> Dict[str, Any]:
        label = "START" if not state else "{" + ", ".join(sorted(state)) + "}"
        return {"label": label, "activities_completed": len(state)}
