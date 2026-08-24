from agentmemorygym_verl.finalizer import _route_ids


def test_route_ids_accepts_canonical_subset_cardinalities():
    assert _route_ids(("swesmith",)) == ("swesmith",)
    assert _route_ids(("webshop", "swesmith", "literesearcher", "openmle_fast")) == (
        "webshop",
        "swesmith",
        "literesearcher",
        "openmle_fast",
    )


def test_route_ids_rejects_empty_duplicate_and_oversized_sets():
    assert _route_ids(()) is None
    assert _route_ids(("swesmith", "swesmith")) is None
    assert _route_ids(("a", "b", "c", "d", "e")) is None
