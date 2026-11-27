from typing import Any, Dict, List

from .abstract_graph_builder import AbstractGraphBuilder, Step

START_NODE = "START"


class LastActivityBasedGraphBuilder(AbstractGraphBuilder):
    """
    Build a graph where each node is the last activity executed in the prefix.
    Edges represent direct transitions between last activities. The START node
    weighs the total number of cases.
    """

    ROOT_ALWAYS = True

    @property
    def name(self) -> str:
        return "Last-Activity-Based Graph"

    def root(self) -> str:
        return START_NODE

    def walk(self, variant: List[str], allow_loops: bool) -> List[Step]:
        steps = []
        previous = START_NODE
        for activity in variant:
            if previous != activity or allow_loops:
                steps.append((previous, activity, activity))
            previous = activity
        return steps

    def node_attributes(self, state: str) -> Dict[str, Any]:
        if state == START_NODE:
            return {"label": "START", "last_activity": None}
        return {"label": state, "last_activity": state}
