import pandas as pd
import pytest

import lcm

CASE, ACTIVITY, TIME = "case:concept:name", "concept:name", "time:timestamp"


def make_log(traces):
    """Build an event log from {case id: "ABC"}, one event per letter."""
    return pd.DataFrame(
        {
            CASE: case,
            ACTIVITY: activity,
            TIME: pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
        }
        for case, trace in traces.items()
        for minute, activity in enumerate(trace)
    )


def generate(traces, **kwargs):
    return lcm.get_service(make_log(traces)).generate(**kwargs)


def nodes(figure):
    return list(figure.data[0].node.label)


def links(figure):
    """Return the links as a set of (source label, target label, cases)."""
    sankey = figure.data[0]
    labels = sankey.node.label
    return {
        (labels[source], labels[target], value)
        for source, target, value in zip(
            sankey.link.source, sankey.link.target, sankey.link.value
        )
    }


def node_colors(figure):
    return dict(zip(figure.data[0].node.label, figure.data[0].node.color))


def legend(figure):
    return [trace.name for trace in figure.data[1:]]


BRANCHING = {"c1": "AB", "c2": "AB", "c3": "AC"}


@pytest.mark.parametrize(
    "map_type, expected",
    [
        (
            "set_based",
            {
                ("START", "{A}", 3),
                ("{A}", "{A, B}", 2),
                ("{A}", "{A, C}", 1),
            },
        ),
        (
            "sequence_based",
            {
                ("START", "A", 3),
                ("A", "A → B", 2),
                ("A", "A → C", 1),
            },
        ),
        (
            "last_activity_based",
            {
                ("START", "A", 3),
                ("A", "B", 2),
                ("A", "C", 1),
            },
        ),
    ],
)
def test_map_type_decides_the_abstraction(map_type, expected):
    figure, _ = generate(BRANCHING, map_type=map_type)
    assert links(figure) == expected


def test_prefixes_in_a_different_order_share_a_set_based_node_only():
    traces = {"c1": "AB", "c2": "BA"}
    set_figure, _ = generate(traces, map_type="set_based")
    sequence_figure, _ = generate(traces, map_type="sequence_based")
    assert sorted(nodes(set_figure)) == ["START", "{A, B}", "{A}", "{B}"]
    assert sorted(nodes(sequence_figure)) == [
        "A",
        "A → B",
        "B",
        "B → A",
        "START",
    ]


def test_map_type_accepts_the_enum():
    figure, _ = generate(BRANCHING, map_type=lcm.MapType.SEQUENCE_BASED)
    assert "A → B" in nodes(figure)


def test_unknown_map_type_is_rejected():
    with pytest.raises(ValueError, match="tree"):
        generate(BRANCHING, map_type="tree")


def test_defaults_draw_a_set_based_map_of_all_activities():
    figure, _ = generate(BRANCHING)
    assert sorted(nodes(figure)) == ["START", "{A, B}", "{A, C}", "{A}"]


def test_metadata_describes_the_map():
    _, metadata = generate(BRANCHING)
    counts = {key: metadata[key] for key in metadata if not key.startswith("traces")}
    assert counts == {
        "nodes_leakage_gt_0": 2,
        "terminal_nodes": 2,
        "total_nodes": 4,
        "empty_cases": 0,
    }


def test_activities_filter_events_out_of_traces():
    figure, _ = generate({"c1": "ABC"}, activities=["A", "C"], map_type="sequence_based")
    assert links(figure) == {("START", "A", 1), ("A", "A → C", 1)}


def test_cases_without_selected_activities_become_an_empty_trace():
    figure, metadata = generate(
        {"c1": "AB", "c2": "CC"}, activities=["A", "B"], map_type="sequence_based"
    )
    assert ("START", "EMPTY TRACE", 1) in links(figure)
    assert metadata["empty_cases"] == 1


def test_empty_traces_can_be_left_out_but_are_still_counted():
    figure, metadata = generate(
        {"c1": "AB", "c2": "CC"},
        activities=["A", "B"],
        visualize_empty_cases=False,
        map_type="sequence_based",
    )
    assert "EMPTY TRACE" not in nodes(figure)
    assert metadata["empty_cases"] == 1


@pytest.mark.parametrize(
    "map_type, label",
    [
        ("set_based", "{EMPTY TRACE}"),
        ("sequence_based", "EMPTY TRACE"),
        ("last_activity_based", "EMPTY TRACE"),
    ],
)
def test_empty_trace_node_is_white(map_type, label):
    figure, _ = generate(
        {"c1": "AB", "c2": "CC"}, activities=["A", "B"], map_type=map_type
    )
    assert node_colors(figure)[label] == "white"


def test_folded_activities_share_their_fold_name():
    figure, _ = generate(
        {"c1": "ABC"}, folds={"F": ["A", "B"]}, map_type="sequence_based"
    )
    assert links(figure) == {
        ("START", "F", 1),
        ("F", "F → F", 1),
        ("F → F", "F → F → C", 1),
    }


def test_consecutive_folded_activities_form_a_self_loop_unless_disallowed():
    with_loops, _ = generate({"c1": "AB"}, folds={"F": ["A", "B"]})
    without_loops, _ = generate(
        {"c1": "AB"}, folds={"F": ["A", "B"]}, allow_loops=False
    )
    assert ("{F}", "{F}", 1) in links(with_loops)
    assert links(without_loops) == {("START", "{F}", 1)}


def test_legend_lists_fold_names_instead_of_their_members():
    figure, _ = generate({"c1": "ABC"}, folds={"F": ["A", "B"]})
    assert legend(figure) == ["C", "F", "EMPTY TRACE"]


def test_activities_are_named_as_they_appear_after_folding():
    figure, metadata = generate(
        {"c1": "AB", "c2": "CC"},
        activities=["F"],
        folds={"F": ["A", "B"]},
        visualize_empty_cases=False,
    )
    assert sorted(nodes(figure)) == ["START", "{F}"]
    assert metadata["empty_cases"] == 1


def test_a_fold_may_reuse_the_name_of_one_of_its_members():
    figure, _ = generate(
        {"c1": "AB"}, folds={"A": ["A", "B"]}, map_type="last_activity_based"
    )
    assert links(figure) == {("START", "A", 1), ("A", "A", 1)}


def test_activities_can_be_listed_as_they_appear_after_folding():
    service = lcm.get_service(make_log({"c1": "ABC"}))
    assert service.get_all_activities(folds={"F": ["A", "B"]}) == ["C", "F"]


def test_listing_activities_rejects_invalid_folds():
    service = lcm.get_service(make_log({"c1": "ABC"}))
    with pytest.raises(ValueError, match="Z"):
        service.get_all_activities(folds={"F": ["Z"]})


def test_folds_apply_only_to_the_call_they_are_passed_to():
    service = lcm.get_service(make_log({"c1": "AB"}))
    service.generate(folds={"F": ["A", "B"]})
    figure, _ = service.generate(map_type="sequence_based")
    assert links(figure) == {("START", "A", 1), ("A", "A → B", 1)}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"activities": []}, "at least one"),
        ({"activities": ["A", "Z"]}, "Z"),
        ({"activities": ["A"], "folds": {"F": ["A", "B"]}}, "A"),
        ({"folds": {"F": ["A", "Z"]}}, "Z"),
        ({"folds": {"F": []}}, "F"),
        ({"folds": {" ": ["A"]}}, "name"),
        ({"folds": {"F": ["A"], "G": ["A", "B"]}}, "A"),
        ({"folds": {"B": ["A"]}}, "B"),
    ],
    ids=[
        "no activities",
        "unknown activity",
        "activity hidden by a fold",
        "unknown fold member",
        "empty fold",
        "blank fold name",
        "activity in two folds",
        "fold named after another activity",
    ],
)
def test_invalid_choices_are_rejected_naming_the_problem(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate({"c1": "ABC"}, **kwargs)


@pytest.mark.parametrize("column", [CASE, ACTIVITY, TIME])
def test_get_service_rejects_a_log_missing_a_column(column):
    with pytest.raises(ValueError, match=column):
        lcm.get_service(make_log({"c1": "AB"}).drop(columns=column))


def test_get_service_leaves_the_log_untouched():
    log = make_log({"c1": "BA"})
    original = log.copy()
    lcm.get_service(log).generate(folds={"F": ["A"]})
    pd.testing.assert_frame_equal(log, original)


def test_events_are_ordered_by_timestamp_not_row_position():
    log = make_log({"c1": "AB"}).iloc[::-1].reset_index(drop=True)
    figure, _ = lcm.get_service(log).generate(map_type="sequence_based")
    assert links(figure) == {("START", "A", 1), ("A", "A → B", 1)}


def test_events_with_equal_timestamps_keep_their_log_order():
    log = make_log({"c1": "BA"})
    log[TIME] = pd.Timestamp("2024-01-01")
    figure, _ = lcm.get_service(log).generate(map_type="sequence_based")
    assert links(figure) == {("START", "B", 1), ("B", "B → A", 1)}


def test_events_without_an_activity_are_ignored():
    log = make_log({"c1": "AB"})
    log.loc[0, ACTIVITY] = None
    service = lcm.get_service(log)
    figure, _ = service.generate(map_type="sequence_based")
    assert service.get_all_activities() == ["B"]
    assert links(figure) == {("START", "B", 1)}


REPEATED = {"c1": "AAAB", "c2": "AAAB"}


@pytest.mark.parametrize(
    "map_type, loop",
    [
        ("set_based", ("{A}", "{A}", 4)),
        ("last_activity_based", ("A", "A", 4)),
    ],
)
def test_self_loops_are_drawn_by_default(map_type, loop):
    figure, _ = generate(REPEATED, map_type=map_type)
    assert loop in links(figure)


@pytest.mark.parametrize(
    "map_type, repeated_node",
    [("set_based", "{A}"), ("last_activity_based", "A")],
)
def test_disallowing_loops_does_not_turn_repeated_activities_into_terminals(
    map_type, repeated_node
):
    figure, metadata = generate(REPEATED, map_type=map_type, allow_loops=False)
    assert node_colors(figure)[repeated_node] != "black"
    assert metadata["terminal_nodes"] == 1


def test_disallowing_loops_changes_nothing_on_sequence_based_maps():
    with_loops, _ = generate(REPEATED, map_type="sequence_based")
    without_loops, _ = generate(REPEATED, map_type="sequence_based", allow_loops=False)
    assert links(with_loops) == links(without_loops)


TOY = {"c1": "AKLZ", "c2": "ALKZ", "c3": "A", "c4": "AKLZ"}
MAP_TYPES = ["set_based", "sequence_based", "last_activity_based"]


def trace_meta(figure):
    return figure.data[0].meta["lcm"]


def link_labels(figure):
    return list(figure.data[0].link.label)


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_traces_are_listed_most_frequent_first(map_type):
    _, metadata = generate(TOY, map_type=map_type)
    assert [t["name"] for t in metadata["traces"]] == [
        "A → K → L → Z",
        "A → L → K → Z",
        "A",
    ]
    assert [t["cases"] for t in metadata["traces"]] == [2, 1, 1]
    assert metadata["traces_truncated"] is False


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_a_trace_names_the_links_it_uses(map_type):
    figure, _ = generate(TOY, map_type=map_type)
    sankey = figure.data[0]
    labels = link_labels(figure)
    for trace in trace_meta(figure)["traces"]:
        used = trace["links"]
        read = [labels[i].rsplit(" (", 1)[0] for i in used]
        assert read == trace["sequence"]
        for before, after in zip(used, used[1:]):
            assert sankey.link.target[before] == sankey.link.source[after]
        assert trace["nodes"][0] == sankey.link.source[used[0]]
        assert trace["nodes"][1:] == [sankey.link.target[i] for i in used]


def test_set_based_traces_share_the_last_link():
    figure, _ = generate(TOY, map_type="set_based")
    long_traces = trace_meta(figure)["traces"][:2]
    assert long_traces[0]["links"][-1] == long_traces[1]["links"][-1]
    assert long_traces[0]["links"][1] != long_traces[1]["links"][1]


def test_last_activity_trace_uses_the_cycle_link_once():
    figure, _ = generate(TOY, map_type="last_activity_based")
    sankey = figure.data[0]
    names = sankey.node.label
    first, second = trace_meta(figure)["traces"][:2]

    def pairs(trace):
        return [
            (names[sankey.link.source[i]], names[sankey.link.target[i]])
            for i in trace["links"]
        ]

    assert pairs(first).count(("K", "L")) == 1
    assert ("L", "K") not in pairs(first)
    assert pairs(second).count(("L", "K")) == 1
    assert len(first["links"]) == 4


def test_hidden_self_loops_leave_no_link():
    figure, metadata = generate(
        {"c": "AAB"}, map_type="set_based", allow_loops=False
    )
    trace = trace_meta(figure)["traces"][0]
    assert trace["name"] == "A → A → B"
    assert len(trace["links"]) == 2
    assert len(trace["nodes"]) == 3


def test_empty_traces_are_one_trace():
    _, metadata = generate(
        {"c1": "AB", "c2": "CC", "c3": "CD"}, activities=["A", "B"]
    )
    names = [t["name"] for t in metadata["traces"]]
    assert names == ["EMPTY TRACE", "A → B"]
    assert metadata["traces"][0]["cases"] == 2


def test_folded_activities_name_the_trace():
    _, metadata = generate({"c1": "ABC"}, folds={"F": ["A", "B"]})
    assert metadata["traces"][0]["name"] == "F → F → C"
    assert metadata["traces"][0]["sequence"] == ["F", "F", "C"]


def test_traces_are_truncated():
    figure, metadata = generate(TOY, trace_limit=2)
    assert len(metadata["traces"]) == 2
    assert metadata["traces_truncated"] is True
    assert trace_meta(figure)["truncated"] is True
    assert len(trace_meta(figure)["traces"]) == 2


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_figure_meta_matches_metadata(map_type):
    figure, metadata = generate(TOY, map_type=map_type)
    meta = trace_meta(figure)
    assert meta["truncated"] == metadata["traces_truncated"]
    keys = ("name", "sequence", "cases")
    assert [{k: t[k] for k in keys} for t in meta["traces"]] == [
        {k: t[k] for k in keys} for t in metadata["traces"]
    ]


def test_figure_without_traces_has_no_meta():
    import networkx as nx
    from lcm.visualization.sankey_visualizer import SankeyVisualizer

    G = nx.MultiDiGraph()
    G.add_node("a", weight=1, label="a")
    G.add_node("b", weight=1, label="b")
    G.add_edge("a", "b", weight=1, activity="x")
    assert SankeyVisualizer().create_figure(G).data[0].meta is None


def test_trace_name_with_arrow_in_activity():
    log = pd.DataFrame(
        {
            CASE: case,
            ACTIVITY: activity,
            TIME: pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
        }
        for case, trace in {"c1": ["a → b"], "c2": ["a", "b"]}.items()
        for minute, activity in enumerate(trace)
    )
    _, metadata = lcm.get_service(log).generate()
    by_sequence = {tuple(t["sequence"]): t["name"] for t in metadata["traces"]}
    assert by_sequence[("a", "b")] == "a → b"
    assert by_sequence[("a → b",)] == "a → b (2)"
    assert len(set(by_sequence.values())) == 2


def test_traces_survive_an_activity_named_like_the_root():
    figure, metadata = generate(
        {"c1": ["START"], "c2": ["START", "A"]},
        map_type="last_activity_based",
        allow_loops=False,
    )
    traces = trace_meta(figure)["traces"]
    assert traces
    root = figure.data[0].node.label.index("START")
    assert all(t["nodes"] and t["nodes"][0] == root for t in traces)


def sample_traces():
    import pathlib

    path = pathlib.Path(__file__).parent.parent / "logs" / "log.csv"
    return lcm.get_service(lcm.load_event_log(str(path)))


def positions_of(figure):
    node = figure.data[0].node
    return list(zip(node.x, node.y))


def node_heights(figure):
    """The height share of each node, as the layout assumed it."""
    from lcm.visualization import trace_layout

    count, weights, edges, values, plot_height = trace_layout.figure_geometry(figure)
    info = figure.data[0].meta["lcm"]
    return trace_layout.layout(count, info["traces"], weights, edges, plot_height, values)[2]


def top_edges(figure, nodes_):
    placed = positions_of(figure)
    height = node_heights(figure)
    return [placed[i][1] - height[i] / 2 for i in nodes_]


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_main_trace_lies_on_one_row(map_type):
    figure, _ = generate(TOY, map_type=map_type, layout="traces")
    placed = positions_of(figure)
    main = trace_meta(figure)["traces"][0]["nodes"]
    tops = top_edges(figure, main)
    assert max(tops) - min(tops) < 1e-6
    if map_type != "last_activity_based":
        xs = [placed[i][0] for i in main]
        assert xs == sorted(set(xs))


def spread_log():
    return {f"c{i}": "A" + "BCDEFGH"[i % 7] + "BCDEFGH"[(i * 3) % 7] + "Z" * (i % 3)
            for i in range(60)}


def test_relayout_follows_the_figure_height():
    figure, _ = generate(spread_log(), map_type="sequence_based", layout="traces")
    before = positions_of(figure)
    figure.update_layout(height=900, margin=dict(l=110, r=10, t=100, b=140))
    assert lcm.relayout(figure) is figure
    assert positions_of(figure) != before
    main = trace_meta(figure)["traces"][0]["nodes"]
    tops = top_edges(figure, main)
    assert max(tops) - min(tops) < 1e-6
    placed = positions_of(figure)
    height = node_heights(figure)
    columns = {}
    for (x, y), h in zip(placed, height):
        columns.setdefault(round(x, 6), []).append((y - h / 2, y + h / 2))
    for spans in columns.values():
        spans.sort()
        for (_, bottom), (top, _) in zip(spans, spans[1:]):
            assert top >= bottom - 1e-9
    assert all(0.001 <= v <= 0.999 for point in placed for v in point)
    assert figure.data[0].node.pad == 10


def test_relayout_leaves_an_auto_figure_alone():
    figure, _ = generate(TOY)
    before = figure.to_json()
    figure.update_layout(height=900)
    expected = figure.to_json()
    assert lcm.relayout(figure).to_json() == expected
    assert before != expected
    assert figure.data[0].node.x is None


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_metadata_is_json_serialisable(map_type):
    import json

    figure, metadata = generate(TOY, map_type=map_type)
    assert json.loads(json.dumps(metadata))["traces"] == trace_meta(figure)["traces"]
    assert set(metadata["traces"][0]) == {"name", "sequence", "cases", "links", "nodes"}


def test_looped_map_keeps_room_above_the_main_row():
    figure, _ = generate(TOY, map_type="last_activity_based", layout="traces")
    root = trace_meta(figure)["traces"][0]["nodes"][0]
    assert top_edges(figure, [root])[0] >= 0.25 - 1e-9


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_no_two_nodes_share_a_position(map_type):
    figure, _ = generate(TOY, map_type=map_type, layout="traces")
    placed = [(round(x, 6), round(y, 6)) for x, y in positions_of(figure)]
    assert len(set(placed)) == len(placed)


def test_auto_layout_sets_no_positions():
    figure, _ = generate(TOY)
    assert figure.data[0].node.x is None
    assert figure.data[0].arrangement not in ("fixed", "freeform")


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="grid"):
        generate(TOY, layout="grid")


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_positions_stay_inside_the_figure(map_type):
    service = sample_traces()
    figure, _ = service.generate(map_type=map_type, layout="traces")
    placed = positions_of(figure)
    assert len(set(placed)) == len(placed)
    assert all(0.001 <= v <= 0.999 for point in placed for v in point)
    assert figure.data[0].arrangement == "freeform"


def test_truncated_map_still_places_every_node():
    figure, metadata = generate(
        TOY, map_type="sequence_based", layout="traces", trace_limit=1
    )
    assert metadata["traces_truncated"] is True
    placed = positions_of(figure)
    assert len(placed) == len(nodes(figure)) == 8
    assert len(set(placed)) == 8


def test_trace_limit_stores_fewer_traces():
    figure, metadata = generate(
        TOY, map_type="sequence_based", layout="traces", trace_limit=2
    )
    assert [t["cases"] for t in metadata["traces"]] == [2, 1]
    assert metadata["traces_truncated"] is True
    assert len(trace_meta(figure)["traces"]) == 2
    placed = positions_of(figure)
    assert len(placed) == len(nodes(figure))
    assert len(set(placed)) == len(placed)


def test_trace_limit_zero_stores_none():
    figure, metadata = generate(TOY, layout="traces", trace_limit=0)
    assert metadata["traces"] == []
    assert metadata["traces_truncated"] is True
    assert "traces" not in trace_meta(figure)
    assert figure.data[0].node.x is None


def test_negative_trace_limit_is_rejected():
    with pytest.raises(ValueError, match="trace_limit"):
        generate(TOY, trace_limit=-1)


def node_tops_from_svg(figure):
    """The top edge in pixels of every node, as Plotly draws it."""
    import re

    try:
        svg = figure.to_image(format="svg").decode()
    except Exception as error:  # no kaleido, or no browser for it
        pytest.skip(f"cannot render the figure: {error}")
    nodes_ = re.findall(
        r'<g class="sankey-node"[^>]*transform="translate\(([-\d.]+),([-\d.]+)\)"',
        svg,
    )
    assert len(nodes_) == len(figure.data[0].node.label)
    return [float(y) for _, y in nodes_]


@pytest.mark.parametrize("map_type", ["set_based", "last_activity_based"])
def test_main_trace_top_edges_agree_in_the_rendered_figure(map_type):
    figure, _ = generate(TOY, map_type=map_type, layout="traces")
    tops = node_tops_from_svg(figure)
    main = trace_meta(figure)["traces"][0]["nodes"]
    drawn = [tops[i] for i in main]
    assert max(drawn) - min(drawn) < 12
