"""Name a LAMMPS potential file's format and print its pair_style lines.

    pair_style_for.py W_zhou.eam.alloy
    pair_style_for.py W_zhou.eam.alloy --json
    pair_style_for.py W.yace --elements W

The header decides the format, and the extension is checked against it.
A setfl file (DYNAMO 86, ``.eam.alloy``) opens with three comment lines,
then ``N El1 El2 ...``, then the grid line ``nrho drho nr dr cutoff``. A
funcfl file (``.eam``) opens with one comment line, then ``Z mass a
lattice``, then the grid line. ``.eam.fs`` files share the setfl layout
under ``eam/fs``. ACE (``.yace``, ``.ace``) and GRACE checkpoints carry no
element symbols in a form worth parsing, so ``--elements`` names them.

The output is the ``pair_style`` and ``pair_coeff`` lines to paste into
the input, and the format name. Exit code 2 means the file matched no
known format; report that rather than editing the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HEAD_LINES = 12


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _is_symbol(token: str) -> bool:
    return 1 <= len(token) <= 2 and token[0].isupper() and token.isalpha()


def _grid_line(tokens: list[str]) -> bool:
    """``nrho drho nr dr cutoff``: int float int float float."""
    return (
        len(tokens) == 5
        and _is_int(tokens[0])
        and _is_float(tokens[1])
        and _is_int(tokens[2])
        and all(_is_float(t) for t in tokens[3:])
    )


def _element_line(tokens: list[str]) -> bool:
    """``Z mass a lattice``: int float float word."""
    return (
        len(tokens) == 4
        and _is_int(tokens[0])
        and _is_float(tokens[1])
        and _is_float(tokens[2])
        and not _is_float(tokens[3])
    )


def classify(path: Path, elements: list[str] | None = None) -> dict[str, Any]:
    """The format, the pair lines, and any mismatch between header and name."""
    name = path.name.lower()
    suffixes = "".join(path.suffixes).lower()
    if path.is_dir() or "grace" in name:
        return _result("grace", "grace", path, elements or [], header_elements=False)
    if suffixes.endswith((".yace", ".ace")):
        return _result("ace", "pace", path, elements or [], header_elements=False)
    if suffixes.endswith(".meam") or name == "library.meam":
        return {
            "format": "meam",
            "pair_style": "meam",
            "pair_coeff": "pair_coeff * * library.meam El... El.meam El...",
            "elements": elements or [],
            "note": "meam needs the library file and the parameter file together",
            "warnings": [],
        }
    try:
        with path.open("rb") as handle:
            head = [
                handle.readline().decode("utf-8", errors="replace").rstrip("\n")
                for _ in range(_HEAD_LINES)
            ]
    except OSError as e:
        return {"format": None, "error": str(e)}
    rows = [line.split() for line in head]
    warnings: list[str] = []
    # setfl: three comments, element line, grid line, then one 'Z mass a lattice' per element.
    if len(rows) > 4 and rows[3] and _is_int(rows[3][0]) and _grid_line(rows[4]):
        count = int(rows[3][0])
        symbols = rows[3][1:]
        if len(symbols) != count or not all(_is_symbol(s) for s in symbols):
            warnings.append(f"element line {rows[3]!r} does not list {count} symbols")
        style = "eam/fs" if suffixes.endswith(".eam.fs") else "eam/alloy"
        if not suffixes.endswith((".eam.alloy", ".eam.fs", ".setfl")):
            warnings.append(f"header is setfl but the name {path.name!r} does not say so")
        return _result(
            "setfl", style, path, elements or symbols, header_elements=True, warnings=warnings
        )
    # funcfl: one comment, 'Z mass a lattice', grid line.
    if len(rows) > 2 and _element_line(rows[1]) and _grid_line(rows[2]):
        if suffixes.endswith((".eam.alloy", ".eam.fs")):
            warnings.append(f"header is funcfl but the name {path.name!r} says setfl")
        return {
            "format": "funcfl",
            "pair_style": "eam",
            "pair_coeff": f"pair_coeff * * {path}",
            "elements": elements or [],
            "note": "funcfl holds one element; for several, one pair_coeff i i FILE per type",
            "warnings": warnings,
        }
    return {"format": None, "error": "the header matches no known format", "head": head[:6]}


def _result(
    form: str,
    style: str,
    path: Path,
    elements: list[str],
    *,
    header_elements: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    if not elements:
        warnings.append("no element symbols known; pass --elements in atom-type order")
    symbols = " ".join(elements) if elements else "El..."
    return {
        "format": form,
        "pair_style": style,
        "pair_coeff": f"pair_coeff * * {path} {symbols}",
        "elements": elements,
        "elements_from_header": header_elements and bool(elements),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="the potential file (or GRACE directory)")
    parser.add_argument(
        "--elements", nargs="+", help="element symbols in atom-type order (overrides the header)"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if not args.file.exists():
        print(f"error: {args.file} does not exist", file=sys.stderr)
        return 2
    result = classify(args.file, args.elements)
    if result.get("format") is None:
        if args.json:
            print(json.dumps(result, indent=1))
        else:
            print(f"error: {result.get('error')}", file=sys.stderr)
            for line in result.get("head", []):
                print(f"  | {line[:100]}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    elements = " ".join(result["elements"]) or "(none in header)"
    print(f"format: {result['format']}  elements: {elements}")
    print(f"pair_style {result['pair_style']}")
    print(result["pair_coeff"])
    if result.get("note"):
        print(f"note: {result['note']}")
    for warning in result.get("warnings", []):
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
