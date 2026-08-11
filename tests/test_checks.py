import pytest

from slab import converged, finite, units, within_bounds


def test_converged_pass_and_fail() -> None:
    good = converged(0.031, below=0.05, label="fmax")
    assert good.passed and good.kind == "converged"
    assert good.message == "fmax=0.031 < 0.05"
    bad = converged(0.12, below=0.05, label="fmax")
    assert not bad.passed
    assert bad.message == "fmax=0.12 !< 0.05"


def test_converged_boundary_is_strict() -> None:
    assert not converged(0.05, below=0.05).passed


@pytest.mark.parametrize("weird", [None, "0.03", float("nan"), float("inf"), [0.03], True])
def test_converged_never_raises_on_weird_observed(weird: object) -> None:
    result = converged(weird, below=0.05)
    assert not result.passed
    assert "not a finite number" in result.message


def test_converged_rejects_non_numeric_threshold() -> None:
    with pytest.raises(TypeError, match="below"):
        converged(0.01, below="0.05")  # type: ignore[arg-type]


def test_within_bounds() -> None:
    ok = within_bounds(-10.8, lo=-12, hi=-9, label="energy")
    assert ok.passed
    assert ok.message == "energy=-10.8 within [-12, -9]"
    assert not within_bounds(3.2, hi=2.0).passed
    assert within_bounds(5, lo=5).passed  # inclusive
    assert within_bounds(1e9, lo=0).passed  # one-sided


def test_within_bounds_requires_a_bound() -> None:
    with pytest.raises(ValueError, match="at least one"):
        within_bounds(1.0)


def test_within_bounds_weird_observed_fails_gracefully() -> None:
    assert not within_bounds(None, lo=0).passed


def test_finite_scalar_and_sequence() -> None:
    assert finite(1.5).passed
    assert finite([1.0, 2.5, -3.0], label="forces").message == "all 3 forces values finite"
    assert not finite(float("inf")).passed
    bad = finite([1.0, float("nan"), 2.0], label="forces")
    assert not bad.passed
    assert "1/3 forces values not finite" in bad.message


def test_finite_rejects_non_numeric_and_empty() -> None:
    assert not finite("100").passed  # strings are not numbers
    assert not finite([]).passed
    assert not finite({"a": 1}).passed  # dicts are ambiguous; fail loudly


def test_finite_flattens_2d_force_arrays() -> None:
    np = pytest.importorskip("numpy")
    forces = np.array([[0.01, 0.02, 0.03], [0.0, 0.01, 0.02]])
    result = finite(forces, label="forces")
    assert result.passed
    assert result.message == "all 6 forces values finite"
    poisoned = forces.copy()
    poisoned[1, 2] = np.nan
    assert not finite(poisoned, label="forces").passed


def test_numpy_scalars_are_numbers() -> None:
    np = pytest.importorskip("numpy")
    assert converged(np.float32(0.01), below=0.05).passed
    assert converged(np.float64(0.12), below=0.05).passed is False  # judged, not rejected
    assert within_bounds(np.int64(5), lo=0, hi=10).passed
    assert finite(np.float32(1.5)).passed
    assert not finite(np.float64("nan")).passed
    # numpy bools are still not numbers
    assert not converged(np.True_, below=0.05).passed


def test_units_annotation_check() -> None:
    assert units("eV", "eV").passed
    assert units("  eV ", "eV").passed  # whitespace-insensitive
    mismatch = units("kcal/mol", "eV")
    assert not mismatch.passed
    assert mismatch.message == "unit 'kcal/mol' != expected 'eV'"
    assert not units("ev", "eV").passed  # case matters
    assert not units(1.0, "eV").passed  # undeclared unit is a failed claim
