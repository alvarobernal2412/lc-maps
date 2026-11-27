import json

import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from lcm.constants import Layout, MapType
from lcm.data.data_processor import DataProcessor
from lcm.services.graph_service import GraphService
from lcm.domain import traces as trace_module
from lcm.domain.graph_analyzer import GraphAnalyzer
from lcm.visualization.sankey_visualizer import SankeyVisualizer


class ProcessAnalyticsService:
    """Orchestrates the analytics pipeline"""

    def __init__(
        self,
        event_log: pd.DataFrame,
        data_processor: DataProcessor,
        graph_service: GraphService,
        graph_analyzer: GraphAnalyzer,
    ):
        self.data_processor = data_processor
        self.graph_service = graph_service
        self.graph_analyzer = graph_analyzer

        self.traces = self.data_processor.prepare_traces(event_log)

    def get_all_activities(
        self, folds: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """List the activities of the log, as they appear after `folds`.

        Raises:
            ValueError: If `folds` are invalid.
        """
        folds = folds or {}
        known = {act for trace in self.traces for act in trace}
        folded = set()
        for name, members in folds.items():
            if not name.strip():
                raise ValueError("Every fold needs a name.")
            if not members:
                raise ValueError(f"Fold '{name}' has no activities.")
            self._reject(set(members) - known, f"Fold '{name}' has unknown activities")
            self._reject(set(members) & folded, f"Fold '{name}' reuses folded activities")
            folded |= set(members)
            if name in known - set(members):
                raise ValueError(f"Fold '{name}' has the name of another activity.")

        return sorted((known - folded) | set(folds))

    def generate(
        self,
        *,
        activities: Optional[List[str]] = None,
        map_type: str = MapType.SET_BASED,
        allow_loops: bool = True,
        visualize_empty_cases: bool = True,
        folds: Optional[Dict[str, List[str]]] = None,
        layout: str = Layout.AUTO,
        trace_limit: int = trace_module.TRACE_LIMIT,
    ) -> Tuple[go.Figure, dict]:
        """Draw a process map of the log.

        Every call starts from the full log, so maps do not affect each other.

        Args:
            activities: Activities to keep, named as they appear after folding.
                All of them if None.
            map_type: A `MapType`, or its value such as "set_based".
            allow_loops: Draw an activity that leaves the node unchanged as a
                self-loop. Has no effect on sequence-based maps, whose nodes
                never repeat.
            visualize_empty_cases: Draw the cases that have none of the selected
                activities as an "EMPTY TRACE" node.
            folds: Fold name mapped to the activities it replaces, for example
                {"Rejections": ["Rejected by A", "Rejected by B"]}.
            layout: "auto" lets Plotly place the nodes; "traces" puts the most
                frequent trace on one row, along the root's. Without stored
                traces (`trace_limit=0`) it falls back to "auto".
            trace_limit: How many of the most frequent traces are stored, in the
                figure and in the metadata. Each one adds to the size of the
                figure; 0 stores none.

        Returns:
            The Plotly Sankey figure, and a dict describing the map: counts,
            the `traces` (most cases first, at most `trace_limit`) and whether they were
            cut (`traces_truncated`). The figure carries the traces too, in
            `figure.data[0].meta["lcm"]`.

        Raises:
            ValueError: If `map_type`, `activities`, `folds` or `layout` or
                `trace_limit` (negative) are invalid.
        """
        map_type = MapType(map_type)
        layout = Layout(layout)
        if trace_limit < 0:
            raise ValueError("trace_limit cannot be negative.")
        folds = folds or {}
        selected_activities = self._select_activities(
            activities, self.get_all_activities(folds)
        )

        traces = self.data_processor.apply_folds(self.traces, folds)
        variants, empty_cases = self.data_processor.create_variants(
            traces, selected_activities, visualize_empty_cases
        )

        G = self.graph_service.build_graph(variants, map_type, allow_loops=allow_loops)
        found, truncated = trace_module.find_traces(
            variants,
            self.graph_service.graph_builders[map_type],
            allow_loops,
            limit=trace_limit,
        )
        G = self.graph_analyzer.enrich_graph_with_metrics(G)

        # Create visualizer with selected activities for legend
        visualizer = SankeyVisualizer(all_activities=selected_activities)
        if trace_limit:
            figure = visualizer.create_figure(
                G, found, truncated=truncated, layout=layout
            )
        else:
            figure = visualizer.create_figure(G, layout=Layout.AUTO)
            if figure.data:
                figure.data[0].meta = {
                    "lcm": {"truncated": truncated, "layout": Layout.AUTO.value}
                }
        # The figure's traces are plain data, the ones the metadata carries.
        listed = (
            json.loads(json.dumps(figure.data[0].meta["lcm"].get("traces", [])))
            if figure.data
            else []
        )
        return figure, {
            "nodes_leakage_gt_0": G.graph.get("leakage_gt_0_count", 0),
            "terminal_nodes": G.graph.get("terminal_count", 0),
            "total_nodes": G.number_of_nodes(),
            "empty_cases": empty_cases,
            "traces": listed,
            "traces_truncated": truncated,
        }

    def _select_activities(
        self, activities: Optional[List[str]], available: List[str]
    ) -> List[str]:
        """Check the chosen activities; return the ones to draw."""
        if activities is None:
            return available
        if not activities:
            raise ValueError("Select at least one activity.")
        self._reject(
            set(activities) - set(available),
            "Unknown activities (folded ones go by fold name)",
        )
        return list(activities)

    @staticmethod
    def _reject(names: set, problem: str) -> None:
        if names:
            raise ValueError(f"{problem}: {', '.join(sorted(names))}.")

    @staticmethod
    def summarize_metadata(metadata: dict) -> dict:
        """Summary of graph metadata for display."""
        return {
            "Total Nodes": metadata["total_nodes"],
            "Nodes with Leakage > 0": metadata["nodes_leakage_gt_0"],
            "Terminal Nodes": metadata["terminal_nodes"],
            "Empty Cases": metadata["empty_cases"],
        }
