"""Arrhenius, VFT, MYEGA, and zero-crossing fits for rate tables, with errors.

    fit_rates.py rates.json --mode arrhenius
    fit_rates.py rates.json --mode arrhenius --jump-length 2.5 --value-unit cm2/s
    fit_rates.py growth.json --mode vft --json
    fit_rates.py growth.json --mode myega
    fit_rates.py velocity.json --mode crossing --window 60

The input is a JSON list of ``{"T": kelvin, "value": rate}`` rows, with an
optional ``"err"`` (one standard deviation of the value, same unit).
Repeated temperatures are replicas and stay in the fit. The value's unit
is yours (a diffusivity, a growth velocity, an escape rate); the fitted
prefactor comes back in the same unit.

* ``arrhenius`` fits ln(value) against 1/T by weighted least squares
  (weights 1/sigma_ln from ``err``; unweighted without) and reports Ea
  and ln A with standard errors and their correlation (near +1: a higher
  barrier comes with a higher prefactor), the rate at the reference
  temperature T_ref (the harmonic mean of the temperatures, where the
  parameters decorrelate), and a curvature verdict from an F-test of a
  quadratic term in 1/T. With ``--jump-length`` (Å),
  ``--dim``, and ``--value-unit`` the prefactor of a diffusivity becomes
  an effective attempt frequency nu = 2 d D0 / a^2.
* ``vft`` fits value = A exp(-B / (T - T0)) with T0 on a grid, and gives
  bootstrap errors for B, T0, and the strength D = B / T0.
* ``myega`` fits value = A exp(-(K / T) exp(C / T)), which has no finite
  divergence temperature and extrapolates to low temperature more safely
  than VFT.
* ``crossing`` locates a sign change, then fits value = k (T_m - T) over
  the rows within ``--window`` kelvin of it and reports T_m with a
  bootstrap error and the kinetic coefficient k.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_KB_EV = 8.617333262e-5  # eV/K
_EV_TO_J = 1.602176634e-19
_BOOTSTRAP_DRAWS = 300
_SEED = 0


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows sorted by temperature, as (T, value, err) arrays; err is NaN where absent."""
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit('error: expected a JSON list of {"T": ..., "value": ...} rows')
    try:
        triples = sorted(
            (float(r["T"]), float(r["value"]), float(r["err"]) if "err" in r else math.nan)
            for r in rows
        )
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: every row needs numeric "T" and "value" fields: {e}') from e
    temperatures = np.array([t for t, _, _ in triples])
    values = np.array([v for _, v, _ in triples])
    errors = np.array([e for _, _, e in triples])
    if (temperatures <= 0).any():
        raise SystemExit("error: temperatures are in kelvin and must be positive")
    if np.isfinite(errors).any() and (errors[np.isfinite(errors)] <= 0).any():
        raise SystemExit('error: "err" is one standard deviation and must be positive')
    if len(np.unique(temperatures)) < 2:
        raise SystemExit("error: at least 2 distinct temperatures are needed")
    return temperatures, values, errors


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 if total == 0.0 else 1.0 - residual / total


def _log_weights(values: np.ndarray, errors: np.ndarray) -> np.ndarray | None:
    """1/sigma in ln space from the value errors, or None when none are given."""
    if not np.isfinite(errors).all():
        return None
    return np.asarray(values / errors)


def _linear(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None
) -> tuple[float, float, np.ndarray | None]:
    """Slope, intercept, and their covariance (None when too few points)."""
    kwargs: dict[str, Any] = {}
    if weights is not None:
        kwargs["w"] = weights
    if len(x) >= 4 or (weights is not None and len(x) >= 3):
        kwargs["cov"] = "unscaled" if weights is not None else True
        (slope, intercept), cov = np.polyfit(x, y, 1, **kwargs)
        return float(slope), float(intercept), np.asarray(cov)
    slope, intercept = np.polyfit(x, y, 1, **kwargs)
    return float(slope), float(intercept), None


def _curvature_test(sse_linear: float, sse_quadratic: float, n: int) -> dict[str, Any] | None:
    """An F-test of one extra parameter (the quadratic term in 1/T).

    Returns None when there are no residual degrees of freedom. Exact data
    (both residuals at rounding level) is called linear.
    """
    dof = n - 3
    if dof <= 0:
        return None
    if sse_linear <= 1e-20:
        return {"f": 0.0, "p": 1.0, "verdict": "arrhenius"}
    from scipy.stats import f as f_distribution  # type: ignore[import-untyped]

    f_value = ((sse_linear - sse_quadratic) / 1.0) / max(sse_quadratic / dof, 1e-300)
    p_value = float(f_distribution.sf(f_value, 1, dof))
    return {
        "f": float(f_value),
        "p": p_value,
        "verdict": "curved" if p_value < 0.05 else "arrhenius",
    }


def _fit_arrhenius(
    temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    if len(np.unique(temperatures)) < 3:
        raise SystemExit("error: an Arrhenius fit needs at least 3 distinct temperatures")
    if (values <= 0).any():
        raise SystemExit(
            "error: Arrhenius fits ln(value); every value must be positive "
            "(fit a growth velocity's magnitude on one side of the crossing only)"
        )
    x = 1.0 / temperatures
    y = np.log(values)
    weights = _log_weights(values, errors)
    slope, intercept, cov = _linear(x, y, weights)
    ea_ev = -slope * _KB_EV
    t_ref = float(1.0 / np.mean(x))
    ln_k_ref = intercept + slope / t_ref
    result: dict[str, Any] = {
        "mode": "arrhenius",
        "ea_eV": ea_ev,
        "ea_J": ea_ev * _EV_TO_J,
        "prefactor": float(math.exp(intercept)),
        "ln_prefactor": intercept,
        "t_ref_K": t_ref,
        "rate_at_t_ref": float(math.exp(ln_k_ref)),
        "r2": _r_squared(y, slope * x + intercept),
        "points": len(temperatures),
        "temperatures": len(np.unique(temperatures)),
        "weighted": weights is not None,
        "warnings": [],
    }
    if cov is not None:
        se_slope, se_intercept = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
        result["ea_se_eV"] = se_slope * _KB_EV
        result["ln_prefactor_se"] = se_intercept
        result["ea_lnA_correlation"] = float(-cov[0, 1] / (se_slope * se_intercept))
        var_ref = cov[1, 1] + cov[0, 0] / t_ref**2 + 2 * cov[0, 1] / t_ref
        result["rate_at_t_ref_se"] = float(math.exp(ln_k_ref) * math.sqrt(max(var_ref, 0.0)))
    else:
        result["ea_se_eV"] = None
        result["ln_prefactor_se"] = None
        result["ea_lnA_correlation"] = None
        result["rate_at_t_ref_se"] = None
        result["warnings"].append(
            "no error bar: three points without errors leave no degrees of freedom; "
            "add temperatures or per-row errors"
        )
    if ea_ev < 0:
        result["warnings"].append(
            "negative activation energy: the rate grows on cooling; "
            "an Arrhenius law is the wrong model for this data"
        )
    n = len(x)
    linear_sse = float(((y - (slope * x + intercept)) ** 2).sum())
    if len(np.unique(temperatures)) >= 4:
        quad = np.polyfit(x, y, 2, w=weights) if weights is not None else np.polyfit(x, y, 2)
        quad_sse = float(((y - np.polyval(quad, x)) ** 2).sum())
        test = _curvature_test(linear_sse, quad_sse, n)
        if test is not None:
            result["curvature"] = test
            if test["verdict"] == "curved":
                result["warnings"].append(
                    f"ln(value) against 1/T is curved: a quadratic term is significant "
                    f"(F-test p = {test['p']:.3g}); the mechanism changes across the "
                    f"window, or the process is non-Arrhenius (try --mode vft or myega)"
                )
    if args.jump_length is not None:
        a = args.jump_length
        if args.value_unit == "cm2/s":
            nu = 2 * args.dim * result["prefactor"] / (a * 1e-8) ** 2
        else:  # A2/fs
            nu = 2 * args.dim * result["prefactor"] / a**2 * 1e15
        result["attempt_frequency_per_s"] = float(nu)
        result["jump_length_A"] = a
        result["dim"] = args.dim
    return result


def _grid_fit(
    temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray, transform: Any,
    grid: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Best (parameter, slope, intercept, sse) over a grid of the nonlinear parameter."""
    y = np.log(values)
    weights = _log_weights(values, errors)
    best: tuple[float, float, float, float] | None = None
    for p in grid:
        x = transform(temperatures, p)
        slope, intercept, _ = _linear(x, y, weights)
        if slope >= 0:  # the rate must slow toward low temperature
            continue
        sse = float(((y - (slope * x + intercept)) ** 2).sum())
        if best is None or sse < best[3]:
            best = (float(p), slope, intercept, sse)
    return best


def _bootstrap(
    temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray, fit: Any,
) -> list[Any]:
    rng = np.random.default_rng(_SEED)
    n = len(temperatures)
    draws: list[Any] = []
    for _ in range(_BOOTSTRAP_DRAWS):
        pick = rng.integers(0, n, size=n)
        if len(np.unique(temperatures[pick])) < 3:
            continue
        try:
            draws.append(fit(temperatures[pick], values[pick], errors[pick]))
        except (SystemExit, np.linalg.LinAlgError, ValueError):
            continue
    return draws


def _vft_transform(t: np.ndarray, t0: float) -> np.ndarray:
    return np.asarray(1.0 / (t - t0))


def _fit_vft(temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    if len(np.unique(temperatures)) < 4:
        raise SystemExit("error: a VFT fit needs at least 4 distinct temperatures")
    if (values <= 0).any():
        raise SystemExit("error: VFT fits ln(value); every value must be positive")
    t_min = float(temperatures.min())
    grid = np.linspace(0.0, 0.98 * t_min, 400)
    best = _grid_fit(temperatures, values, errors, _vft_transform, grid)
    if best is None:
        raise SystemExit(
            "error: no physical VFT fit (B > 0) exists for this data; "
            "the values do not decrease toward low temperature"
        )
    t0, slope, intercept, _ = best
    x = _vft_transform(temperatures, t0)
    y = np.log(values)

    def refit(t: np.ndarray, v: np.ndarray, e: np.ndarray) -> tuple[float, float]:
        again = _grid_fit(t, v, e, _vft_transform, np.linspace(0.0, 0.98 * float(t.min()), 120))
        if again is None:
            raise ValueError("no fit")
        return again[0], -again[1]

    draws = _bootstrap(temperatures, values, errors, refit)
    result: dict[str, Any] = {
        "mode": "vft",
        "prefactor": float(math.exp(intercept)),
        "b_K": -slope,
        "t0_K": t0,
        "strength_D": (-slope / t0) if t0 > 0 else None,
        "r2": _r_squared(y, slope * x + intercept),
        "points": len(temperatures),
        "warnings": [],
    }
    if len(draws) >= 20:
        t0s = np.array([d[0] for d in draws])
        bs = np.array([d[1] for d in draws])
        result["t0_se_K"] = float(t0s.std(ddof=1))
        result["b_se_K"] = float(bs.std(ddof=1))
        result["b_t0_correlation"] = (
            float(np.corrcoef(bs, t0s)[0, 1]) if t0s.std() > 0 and bs.std() > 0 else None
        )
    else:
        result["t0_se_K"] = result["b_se_K"] = result["b_t0_correlation"] = None
    if t0 >= grid[-2]:
        result["warnings"].append(
            "T0 sits at the edge of its search range: the data does not "
            "constrain the divergence temperature; add lower-temperature points"
        )
    if t0 < 0.02 * t_min:
        result["warnings"].append(
            "best T0 is within 2 % of 0 K: the data is plain Arrhenius, not VFT"
        )
    return result


def _myega_transform(t: np.ndarray, c: float) -> np.ndarray:
    return np.asarray(np.exp(c / t) / t)


def _fit_myega(temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    if len(np.unique(temperatures)) < 4:
        raise SystemExit("error: a MYEGA fit needs at least 4 distinct temperatures")
    if (values <= 0).any():
        raise SystemExit("error: MYEGA fits ln(value); every value must be positive")
    t_max = float(temperatures.max())
    grid = np.linspace(0.0, 3.0 * t_max, 400)
    best = _grid_fit(temperatures, values, errors, _myega_transform, grid)
    if best is None:
        raise SystemExit(
            "error: no physical MYEGA fit (K > 0) exists for this data; "
            "the values do not decrease toward low temperature"
        )
    c, slope, intercept, _ = best
    x = _myega_transform(temperatures, c)
    y = np.log(values)
    result: dict[str, Any] = {
        "mode": "myega",
        "prefactor": float(math.exp(intercept)),
        "k_K": -slope,
        "c_K": c,
        "r2": _r_squared(y, slope * x + intercept),
        "points": len(temperatures),
        "warnings": [],
    }
    if c >= grid[-2]:
        result["warnings"].append(
            "C sits at the edge of its search range; the data does not constrain the "
            "low-temperature curvature"
        )
    if c == 0.0:
        result["warnings"].append("best C is 0 K: the data is plain Arrhenius, not MYEGA")
    return result


def _crossing_points(temperatures: np.ndarray, values: np.ndarray) -> list[dict[str, float]]:
    crossings: list[dict[str, float]] = []
    for i in range(len(temperatures) - 1):
        v1, v2 = float(values[i]), float(values[i + 1])
        t1, t2 = float(temperatures[i]), float(temperatures[i + 1])
        if t2 == t1:
            continue
        slope = (v2 - v1) / (t2 - t1)
        if v1 == 0.0:
            crossings.append({"T_K": t1, "slope_per_K": slope})
        elif v1 * v2 < 0:
            crossings.append({"T_K": t1 - v1 / slope, "slope_per_K": slope})
        if i == len(temperatures) - 2 and v2 == 0.0:
            crossings.append({"T_K": t2, "slope_per_K": slope})
    return crossings


def _find_crossings(
    temperatures: np.ndarray, values: np.ndarray, errors: np.ndarray, window: float
) -> dict[str, Any]:
    if len(temperatures) < 2:
        raise SystemExit("error: a crossing needs at least 2 temperatures")
    crossings = _crossing_points(temperatures, values)
    if not crossings:
        raise SystemExit(
            "error: the values never change sign; extend the temperature "
            "ladder across the transition before asking for a crossing"
        )
    result: dict[str, Any] = {
        "mode": "crossing",
        "crossings": crossings,
        "points": len(temperatures),
        "warnings": [],
    }
    if len(crossings) > 1:
        result["warnings"].append(
            f"{len(crossings)} sign changes: the data is not monotonic around "
            f"the transition; check the individual runs before quoting one"
        )
    guess = crossings[0]["T_K"]
    near = np.abs(temperatures - guess) <= window
    weights = None
    if np.isfinite(errors[near]).all():
        weights = 1.0 / errors[near]
    if near.sum() >= 3 and len(np.unique(temperatures[near])) >= 2:
        slope, intercept, cov = _linear(temperatures[near], values[near], weights)
        if slope >= 0:
            result["warnings"].append(
                "the velocity rises with temperature through the crossing; the sign "
                "convention is reversed (growth positive, melting negative)"
            )
        t_m = -intercept / slope if slope != 0 else guess
        result["t_m_K"] = float(t_m)
        result["kinetic_coefficient_per_K"] = float(-slope)
        result["window_K"] = window
        result["window_points"] = int(near.sum())

        def refit(t: np.ndarray, v: np.ndarray, e: np.ndarray) -> float:
            w = 1.0 / e if np.isfinite(e).all() else None
            s, b, _ = _linear(t, v, w)
            if s == 0:
                raise ValueError("flat")
            return float(-b / s)

        draws = _bootstrap(temperatures[near], values[near], errors[near], refit)
        draws = [d for d in draws if abs(d - t_m) < 5 * window]
        result["t_m_se_K"] = float(np.std(draws, ddof=1)) if len(draws) >= 20 else None
        if cov is not None and slope != 0:
            var = (cov[1, 1] + t_m**2 * cov[0, 0] + 2 * t_m * cov[0, 1]) / slope**2
            result["t_m_se_from_fit_K"] = float(math.sqrt(max(var, 0.0)))
        predicted = slope * temperatures[near] + intercept
        result["window_r2"] = _r_squared(values[near], predicted)
        if result["window_r2"] < 0.9:
            result["warnings"].append(
                "the velocity is not linear across the window; narrow --window or run "
                "closer to the crossing"
            )
    else:
        result["t_m_K"] = float(guess)
        result["t_m_se_K"] = None
        result["kinetic_coefficient_per_K"] = float(-crossings[0]["slope_per_K"])
        result["window_points"] = int(near.sum())
        result["warnings"].append(
            f"fewer than 3 points within {window:g} K of the crossing: T_m is a two-point "
            f"interpolation with no error bar; run more temperatures near it"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "data", type=Path, help='JSON list of {"T": K, "value": rate, "err": optional} rows'
    )
    parser.add_argument(
        "--mode", required=True, choices=("arrhenius", "vft", "myega", "crossing"),
        help="the law to fit (crossing locates a sign change instead)",
    )
    parser.add_argument(
        "--window", type=float, default=60.0,
        help="crossing: half-width in K of the linear-fit window around T_m (default 60)",
    )
    parser.add_argument(
        "--jump-length", type=float,
        help="arrhenius: jump length in A, to turn a diffusion prefactor into an "
        "attempt frequency",
    )
    parser.add_argument("--dim", type=int, default=3, help="dimensionality of the jumps")
    parser.add_argument(
        "--value-unit", choices=("cm2/s", "A2/fs"), default="cm2/s",
        help="unit of a diffusivity value, for the attempt frequency",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if args.window <= 0:
        raise SystemExit("error: --window is a half-width in kelvin and must be positive")
    if args.jump_length is not None and args.jump_length <= 0:
        raise SystemExit("error: --jump-length is a distance in A and must be positive")
    if args.dim not in (1, 2, 3):
        raise SystemExit("error: --dim must be 1, 2, or 3")

    temperatures, values, errors = _load(args.data)
    if args.mode == "arrhenius":
        result = _fit_arrhenius(temperatures, values, errors, args)
    elif args.mode == "vft":
        result = _fit_vft(temperatures, values, errors)
    elif args.mode == "myega":
        result = _fit_myega(temperatures, values, errors)
    else:
        result = _find_crossings(temperatures, values, errors, args.window)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0

    def pm(value: float | None, digits: int = 4) -> str:
        return "" if value is None else f" +/- {value:.{digits}g}"

    if result["mode"] == "arrhenius":
        print(
            f"Arrhenius over {result['points']} points at {result['temperatures']} "
            f"temperatures ({'weighted' if result['weighted'] else 'unweighted'}): "
            f"Ea = {result['ea_eV']:.4f}{pm(result['ea_se_eV'])} eV = {result['ea_J']:.4e} J"
        )
        print(
            f"ln A = {result['ln_prefactor']:.3f}{pm(result['ln_prefactor_se'], 3)} "
            f"(prefactor {result['prefactor']:.4e} in the input unit), R^2 = {result['r2']:.4f}"
        )
        if result["ea_lnA_correlation"] is not None:
            print(f"Ea-lnA correlation {result['ea_lnA_correlation']:+.3f}")
        print(
            f"rate at T_ref = {result['t_ref_K']:.1f} K: {result['rate_at_t_ref']:.4e}"
            f"{pm(result['rate_at_t_ref_se'], 3)} (the decorrelated parametrisation)"
        )
        if "curvature" in result:
            print(f"curvature verdict: {result['curvature']['verdict']}")
        if "attempt_frequency_per_s" in result:
            print(
                f"attempt frequency nu = 2 d D0 / a^2 = "
                f"{result['attempt_frequency_per_s']:.3e} /s (a = {result['jump_length_A']} A, "
                f"d = {result['dim']})"
            )
    elif result["mode"] == "vft":
        d_text = f", D = B/T0 = {result['strength_D']:.2f}" if result["strength_D"] else ""
        print(
            f"VFT over {result['points']} points: "
            f"prefactor = {result['prefactor']:.4e} (input unit), "
            f"B = {result['b_K']:.1f}{pm(result['b_se_K'], 3)} K, "
            f"T0 = {result['t0_K']:.1f}{pm(result['t0_se_K'], 3)} K{d_text}, "
            f"R^2 = {result['r2']:.4f}"
        )
        if result["b_t0_correlation"] is not None:
            print(f"B-T0 bootstrap correlation {result['b_t0_correlation']:+.3f}")
    elif result["mode"] == "myega":
        print(
            f"MYEGA over {result['points']} points: "
            f"prefactor = {result['prefactor']:.4e} (input unit), "
            f"K = {result['k_K']:.1f} K, C = {result['c_K']:.1f} K, R^2 = {result['r2']:.4f}"
        )
    else:
        for crossing in result["crossings"]:
            print(
                f"zero crossing at T = {crossing['T_K']:.1f} K "
                f"(local slope {crossing['slope_per_K']:.4e} per K)"
            )
        print(
            f"T_m = {result['t_m_K']:.1f}{pm(result['t_m_se_K'], 3)} K from "
            f"{result['window_points']} point(s); kinetic coefficient k = "
            f"{result['kinetic_coefficient_per_K']:.4e} per K"
        )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
