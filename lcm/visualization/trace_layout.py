"""Node positions that give the most frequent trace a straight band from the root."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

LOW, HIGH = 0.001, 0.999
# The gap between the nodes of a column, in pixels; also the Sankey `pad`.
PAD_PX = 10
# What Plotly uses when the layout sets no height or margin.
DEFAULT_HEIGHT = 700
DEFAULT_MARGIN = {"t": 100, "b": 80}
DEFAULT_PLOT_HEIGHT = DEFAULT_HEIGHT - DEFAULT_MARGIN["t"] - DEFAULT_MARGIN["b"]


def layout(
    node_count,
    traces,
    weights,
    edges=None,
    plot_height=DEFAULT_PLOT_HEIGHT,
    values=None,
):
    """Return the x and y of every node (centres, each in [0.001, 0.999]) and the
    height of each node as a share of the plot.

    `traces` are the figure's traces (their `nodes` are node indexes, the root
    first), `weights` the height of each node, `edges` the (source, target) of
    the links (`values` their widths, needed to size the nodes of a map with
    cycles as Plotly does) and `plot_height` the height in pixels of the area
    the Sankey fills.

    A node sits in the column where a trace reaches it for the first time, the
    latest one if there are several, except that the nodes of the first trace
    keep their own, so it always runs left to right. A node sits in the row of
    the first trace that reaches it. The nodes of the first trace share the top
    edge of the root; the other nodes of a column are stacked below, by row.
    """
    if node_count == 0:
        return [], [], []
    column, row, order = _columns_and_rows(node_count, traces, edges or [])
    x = _x_of(column)
    first = traces[0]["nodes"] if traces else []
    scale = node_scale(weights, edges or [], values, plot_height, column)
    looped = any(column[t] <= column[s] for s, t in edges or [])
    y, height = _y_of(
        column, row, order, weights, first, PAD_PX / plot_height, scale, looped
    )
    return x, y, height


def figure_geometry(figure):
    """Return the node count, weights, links, link values and plot height of a figure."""
    sankey = figure.data[0]
    count = len(sankey.node.label)
    flow_in, flow_out = [0] * count, [0] * count
    edges = list(zip(sankey.link.source, sankey.link.target))
    for (source, target), value in zip(edges, sankey.link.value):
        flow_out[source] += value
        flow_in[target] += value
    weights = [max(a, b, 1) for a, b in zip(flow_in, flow_out)]
    margin = figure.layout.margin
    top = margin.t if margin.t is not None else DEFAULT_MARGIN["t"]
    bottom = margin.b if margin.b is not None else DEFAULT_MARGIN["b"]
    height = figure.layout.height or DEFAULT_HEIGHT
    return count, weights, edges, list(sankey.link.value), max(height - top - bottom, 1)


def relayout(figure):
    """Place the nodes of a figure drawn with `layout="traces"` again.

    Call it after changing the height or the margins of the figure. A figure
    that was drawn with `layout="auto"`, or has no traces, stays as it is.
    """
    if not figure.data:
        return figure
    meta = figure.data[0].meta
    info = meta.get("lcm") if isinstance(meta, dict) else None
    if not info or info.get("layout") != "traces":
        return figure
    count, weights, edges, values, plot_height = figure_geometry(figure)
    x, y, _ = layout(count, info["traces"], weights, edges, plot_height, values)
    sankey = figure.data[0]
    sankey.arrangement = "freeform"
    sankey.node.x = x
    sankey.node.y = y
    sankey.node.pad = PAD_PX
    return figure


def _columns_and_rows(node_count, traces, edges):
    column: List[Optional[int]] = [None] * node_count
    row: List[Optional[int]] = [None] * node_count
    order: List[int] = []  # nodes in the order the traces place them

    later: Dict[int, int] = {}
    for trace in traces:
        seen = set()
        for position, node in enumerate(trace["nodes"]):
            if node not in seen:
                seen.add(node)
                later[node] = max(later.get(node, 0), position)
    if traces:
        first = {}
        for position, node in enumerate(traces[0]["nodes"]):
            first.setdefault(node, position)
        later.update(first)
    for node, value in later.items():
        column[node] = value

    next_row = 0
    for trace in traces:
        owns_row = False
        for node in trace["nodes"]:
            if row[node] is None:
                row[node] = next_row
                owns_row = True
                order.append(node)
        if owns_row:
            next_row += 1

    # Nodes no stored trace reaches: one column after what leads to them.
    loose = [node for node in range(node_count) if column[node] is None]
    changed = True
    while loose and changed:
        changed = False
        for source, target in edges:
            if column[target] is None and column[source] is not None:
                column[target] = column[source] + 1
                changed = True
    last = max((c for c in column if c is not None), default=0)
    for node in loose:
        if column[node] is None:
            column[node] = last
        row[node] = next_row
        next_row += 1
        order.append(node)
    return column, row, order


def _x_of(column: List[int]) -> List[float]:
    ranks = {value: rank for rank, value in enumerate(sorted(set(column)))}
    widest = max(ranks.values())
    if widest == 0:
        return [LOW] * len(column)
    return [min(HIGH, max(LOW, ranks[value] / widest)) for value in column]


def _y_of(column, row, order, weights, first, pad, scale, looped=False):
    count = len(column)
    rank_in_order = {node: i for i, node in enumerate(order)}
    stacks: Dict[int, List[int]] = defaultdict(list)
    for node in range(count):
        stacks[column[node]].append(node)
    for stack in stacks.values():
        stack.sort(key=lambda node: (row[node], rank_in_order[node]))
    flow = [max(weights[node], 1e-9) for node in range(count)]

    height = [w * scale for w in flow]
    # Plotly's gap can never be kept in a column with more nodes than fit.
    widest = max(len(stack) for stack in stacks.values())
    gap = min(pad, 0.5 / max(1, widest - 1))
    band = min((height[node] for node in first), default=0.0)

    def stacked(stack):
        return sum(height[node] for node in stack) + gap * (len(stack) - 1)

    # The first row centres the tallest column; loops and back links are drawn
    # above the nodes and need room there.
    common = max(LOW, (1 - max(stacked(s) for s in stacks.values())) / 2)
    if looped:
        common = max(common, 0.25)

    y = [0.5] * count
    for stack in stacks.values():
        total = stacked(stack)
        if row[stack[0]] == 0:
            top = common
        else:
            # No node of the first trace here: stay below its band if there is room.
            top = common + band + gap
        # A column that does not fit starts higher, as far as it has to.
        top = max(LOW, min(top, HIGH - total))
        for node in stack:
            y[node] = min(HIGH, max(LOW, top + height[node] / 2))
            top += height[node] + gap
    return y, height


def node_scale(weights, edges, values, plot_height, column):
    """Height share of the plot per unit of node weight, as Plotly sizes the nodes.

    Plotly gives every node the same height per unit of weight: what the fullest
    column leaves after its gaps. A map with cycles or self-loops is drawn by a
    different layout, which shrinks the nodes to 30% and then again to make room
    for the loops that run around the nodes. The positions then refer to the
    smaller area left for the nodes, so the loops' own room does not matter
    here. Which links count as loops is decided by a depth-first search, not by
    Plotly's own search, so cycles may be sized slightly off.
    """
    count = len(weights)
    pad = PAD_PX
    flow = [max(w, 1e-9) for w in weights]
    loops = _circular_links(count, edges) if values is not None else set()
    if not loops:
        columns = _by_column(count, column)
        return min(
            max(plot_height - pad * (len(c) - 1), plot_height * 0.02)
            / sum(flow[n] for n in c)
            for c in columns
        ) / plot_height
    kept = [e for i, e in enumerate(edges) if i not in loops]
    out_degree = [0] * count
    for source, _ in edges:
        out_degree[source] += 1
    depth = _depths(count, kept)
    last = max(depth)
    plotly_column = [depth[n] if out_degree[n] else last for n in range(count)]
    columns = _by_column(count, plotly_column)
    scale = 0.3 * min(
        max(plot_height - pad * (len(c) - 1), plot_height * 0.02)
        / sum(flow[n] for n in c)
        for c in columns
    )
    top = bottom = 0.0
    side: Dict[int, str] = {}
    flip = 0
    for i in sorted(loops):
        source, target = edges[i]
        kind = side.get(source) or side.get(target)
        if kind is None:
            kind = "top" if flip % 2 == 0 else "bottom"
            flip += 1
        side[source] = side[target] = kind
        if kind == "top":
            top += values[i] * scale
        else:
            bottom += values[i] * scale
    top = top + 35 if top else 0
    bottom = bottom + 35 if bottom else 0
    # Plotly squeezes the nodes into what the loops leave at the top and bottom,
    # and the positions are shares of the area plus that room.
    return scale / (plot_height + top + bottom)


def _by_column(count, column):
    groups = defaultdict(list)
    for node in range(count):
        groups[column[node]].append(node)
    return list(groups.values())


def _circular_links(count, edges):
    """Indexes of the self-loops and of the links that close a cycle."""
    loops = {i for i, (s, t) in enumerate(edges) if s == t}
    out = defaultdict(list)
    for i, (s, t) in enumerate(edges):
        if s != t:
            out[s].append((t, i))
    state = [0] * count
    for start in range(count):
        if state[start]:
            continue
        state[start] = 1
        stack = [(start, iter(out[start]))]
        while stack:
            node, links = stack[-1]
            for target, i in links:
                if state[target] == 1:
                    loops.add(i)
                elif state[target] == 0:
                    state[target] = 1
                    stack.append((target, iter(out[target])))
                    break
            else:
                state[node] = 2
                stack.pop()
    return loops


def _depths(count, edges):
    """Longest path from a node without incoming links, ignoring loops."""
    incoming = [0] * count
    out = defaultdict(list)
    for s, t in edges:
        incoming[t] += 1
        out[s].append(t)
    depth = [0] * count
    ready = [n for n in range(count) if not incoming[n]]
    while ready:
        node = ready.pop()
        for target in out[node]:
            depth[target] = max(depth[target], depth[node] + 1)
            incoming[target] -= 1
            if not incoming[target]:
                ready.append(target)
    return depth
