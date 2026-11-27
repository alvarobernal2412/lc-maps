from collections import Counter
from typing import Dict, List, Tuple

from lcm.graph_builders import AbstractGraphBuilder

SEPARATOR = " → "
# The most traces a map keeps; the rest are counted as truncated.
TRACE_LIMIT = 500


def find_traces(
    variants: List[List[str]],
    builder: AbstractGraphBuilder,
    allow_loops: bool,
    limit: int = TRACE_LIMIT,
) -> Tuple[List[dict], bool]:
    """List the distinct variants, the most frequent first.

    Traces with as many cases come longest first, then in sequence order.

    A trace is named by its activities joined with " → ". Two different
    sequences can only get the same name when an activity holds the arrow
    itself; the later one then gets " (2)", " (3)" and so on, so use `sequence`
    to identify a trace, not the name.

    Each trace has its `name`, its `sequence` of activities, the number of
    `cases` that follow it, the `root` state and the `steps` it takes in the map `builder` draws.
    The second value is True when more than `limit` traces exist and the rarest
    were cut.
    """
    counts = Counter(tuple(variant) for variant in variants)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    truncated = len(ordered) > limit
    kept = ordered[:limit]

    names = _unique_names([sequence for sequence, _ in kept])
    traces = [
        {
            "name": name,
            "sequence": list(sequence),
            "cases": cases,
            "root": builder.root(),
            "steps": builder.walk(list(sequence), allow_loops),
        }
        for name, (sequence, cases) in zip(names, kept)
    ]
    return traces, truncated


def _unique_names(sequences: List[Tuple[str, ...]]) -> List[str]:
    """Join each sequence with arrows; a repeated name gets a number."""
    names, used = [], set()
    for sequence in sequences:
        name = SEPARATOR.join(sequence)
        base, number = name, 1
        while name in used:
            number += 1
            name = f"{base} ({number})"
        used.add(name)
        names.append(name)
    return names
