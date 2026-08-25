import pytest

from foundation import SerializationError, dumps, fingerprint, loads


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        42,
        -1.5,
        "text",
        "",
        [1, 2, [3, 4]],
        {"a": 1, "b": {"c": [1.0, None]}},
        (1, 2, 3),
        {1, 2, 3},
        b"raw bytes",
        ("mixed", [1], {"k": 2}),
    ],
)
def test_roundtrip_identity(value: object) -> None:
    restored = loads(dumps(value))
    assert restored == value
    assert type(restored) is type(value)


def test_json_values_are_canonical() -> None:
    assert dumps({"b": 1, "a": 2}) == dumps({"a": 2, "b": 1})
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})


def test_json_faithful_values_use_json_tag() -> None:
    assert dumps({"kpts": [4, 4, 4]}).startswith(b"J\n")
    assert dumps(b"blob").startswith(b"B\n")
    assert dumps((1, 2)).startswith(b"P\n")


def test_tuple_and_list_hash_differently() -> None:
    assert fingerprint((1, 2)) != fingerprint([1, 2])


def test_int_keyed_dict_keeps_fidelity() -> None:
    value = {1: "a", 2: "b"}  # JSON would stringify the keys
    assert loads(dumps(value)) == value


def test_fingerprint_shape_and_stability() -> None:
    a = fingerprint({"x": 1.0})
    assert a == fingerprint({"x": 1.0})
    assert len(a) == 64
    assert fingerprint(1) != fingerprint("1") != fingerprint(1.0)


def test_unserializable_value_raises() -> None:
    with pytest.raises(SerializationError, match="cannot serialize"):
        dumps(lambda x: x)


def test_unknown_tag_raises() -> None:
    with pytest.raises(SerializationError, match="unrecognized"):
        loads(b"Z\nwhatever")


def test_nan_roundtrip() -> None:
    import math

    restored = loads(dumps({"e": float("nan")}))
    assert isinstance(restored, dict) and math.isnan(restored["e"])
