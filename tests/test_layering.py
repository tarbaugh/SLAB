"""The dependency direction between the packages, enforced.

``slab_stack`` may import everything (it is the distribution's umbrella).
``mason`` may import ``foundation`` and ``slab``. ``foundation`` may import
``slab``. ``slab`` imports neither. Nothing in the language stops a stray
``from foundation import ...`` inside ``slab``; this test does.

The check reads the AST rather than the import graph at runtime, so it sees
imports that only execute on some paths: inside functions, inside
``if TYPE_CHECKING:`` blocks, and inside ``try``/``except ImportError``
fallbacks. Those are exactly where a violation hides from a passing suite.
"""

from __future__ import annotations

import ast
import functools
import sys
import tomllib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

# What each package is allowed to import from this repository, itself included.
ALLOWED: dict[str, frozenset[str]] = {
    "slab": frozenset({"slab"}),
    "foundation": frozenset({"foundation", "slab"}),
    "mason": frozenset({"mason", "foundation", "slab"}),
    "slab_stack": frozenset({"slab_stack", "mason", "foundation", "slab"}),
}
PACKAGES = tuple(ALLOWED)


@functools.cache
def _packages_on_disk() -> frozenset[str]:
    """Every importable package under ``src/``, declared or not.

    Cached: the import scan asks once per import statement, and ``src/``
    does not change during a test run.
    """
    return frozenset(
        p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
    )


def _imported_roots(tree: ast.AST) -> set[tuple[str, int]]:
    """Every in-repo package this module names, with the line that names it.

    "In-repo" is read from disk, not from the declared list, so an undeclared
    package still registers as an edge rather than passing as third-party.
    """
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _packages_on_disk():
                    found.add((root, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import cannot leave its own package
                continue
            if node.module is None:
                continue
            root = node.module.split(".", 1)[0]
            if root in _packages_on_disk():
                found.add((root, node.lineno))
    return found


def _modules_of(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports_only_what_it_may(package: str) -> None:
    allowed = ALLOWED[package]
    violations: list[str] = []
    for path in _modules_of(package):
        tree = ast.parse(path.read_text(), filename=str(path))
        for root, lineno in sorted(_imported_roots(tree)):
            if root not in allowed:
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} imports {root!r}")
    assert not violations, (
        f"{package} may import {sorted(allowed)} and nothing else from this "
        f"repository, but:\n  " + "\n  ".join(violations)
    )


def test_every_package_is_actually_scanned() -> None:
    """A typo in SRC or a renamed package would make the rule vacuous."""
    for package in PACKAGES:
        modules = _modules_of(package)
        assert modules, f"no modules found for {package} under {SRC}"
        assert (SRC / package / "__init__.py").is_file()


def test_no_package_escapes_the_rule() -> None:
    """A package nobody declared is a laundering route between the others.

    Without this, adding ``src/thing`` that imports ``mason`` and is imported
    by ``slab`` satisfies every per-package check while carrying a dependency
    the whole point of the rule is to forbid: the scan only ever looked at the
    three names it knew. Declaring a new package in ``ALLOWED`` is what makes
    it visible, so the tuple and the directory must agree.
    """
    on_disk = _packages_on_disk()
    declared = set(ALLOWED)
    assert on_disk == declared, (
        f"src/ holds {sorted(on_disk)} but the layering rule declares "
        f"{sorted(declared)}; add the new package to ALLOWED with the "
        f"packages it may import, and to pyproject's wheel packages"
    )


def test_slab_is_reached_from_foundation_and_mason() -> None:
    """The rule is a ceiling, not a description. Confirm the edges exist."""
    def roots(package: str) -> set[str]:
        seen: set[str] = set()
        for path in _modules_of(package):
            tree = ast.parse(path.read_text(), filename=str(path))
            seen |= {root for root, _ in _imported_roots(tree)}
        return seen

    assert "slab" in roots("foundation")
    assert "foundation" in roots("mason")


@pytest.mark.parametrize("package", PACKAGES)
def test_package_resolves_inside_this_checkout(package: str) -> None:
    module = __import__(package)
    assert module.__file__ is not None
    assert Path(module.__file__).resolve().is_relative_to(SRC), (
        f"{package} resolved to {module.__file__}, outside {SRC}; an installed "
        f"copy is shadowing the working tree"
    )


def test_pyproject_declares_the_front_door_script() -> None:
    """One console script, ``slab``, and it is the front door in
    ``slab_stack`` — the only package allowed to import all the others.
    The per-package apps are internal; nothing else earns an entry point."""
    pyproject = tomllib.loads((SRC.parent / "pyproject.toml").read_text())
    scripts = pyproject["project"]["scripts"]
    assert scripts == {"slab": "slab_stack.cli:app"}


def test_wheel_ships_every_package() -> None:
    pyproject = tomllib.loads((SRC.parent / "pyproject.toml").read_text())
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert sorted(packages) == sorted(f"src/{name}" for name in PACKAGES)


@pytest.mark.parametrize("package", PACKAGES)
def test_package_ships_a_py_typed_marker(package: str) -> None:
    """Without it, a downstream mypy silently ignores this package's types."""
    assert (SRC / package / "py.typed").is_file()


@pytest.mark.parametrize("first,second", [("slab", "foundation"), ("foundation", "slab")])
def test_packages_import_cleanly_in_either_order(first: str, second: str) -> None:
    """A cycle would only show up from a cold interpreter, in one order."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", f"import {first}; import {second}; print('ok')"],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_slab_stays_light() -> None:
    """The heavy imports live behind slab.backends and foundation.tasks."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, slab, foundation; "
            "heavy = [m for m in ('ase', 'numpy', 'torch', 'mace') if m in sys.modules]; "
            "print(','.join(heavy))",
        ],
        capture_output=True,
        text=True,
        cwd=SRC.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing slab and foundation pulled in {result.stdout.strip()}; the "
        f"heavy dependencies must stay behind slab.backends / foundation.tasks"
    )


def test_builtin_cards_and_skills_ship_inside_the_package() -> None:
    """The roster's built-ins are package data: they must live under src/mason
    (hatchling ships whole package directories, the py.typed precedent), and
    every built-in skill must carry its manifest."""
    cards = sorted(p.name for p in (SRC / "mason" / "agents").glob("*.md"))
    assert cards == ["analysis-expert.md", "dft-expert.md", "md-expert.md", "pi.md"]
    skills = sorted(
        p.name for p in (SRC / "mason" / "skills").iterdir() if p.is_dir()
    )
    assert skills == [
        "atomsk-defects",
        "atomsk-interfaces",
        "atomsk-structures",
        "convergence-study",
        "elastic-constants",
        "equation-of-state",
        "interface-adhesion",
        "kinetic-fits",
        "melt-quench",
        "msd-diffusion",
        "nemd-transport",
        "nucleation-cnt",
        "radial-distribution",
        "surface-energy",
        "thermal-response",
        "two-phase-melting",
    ]
    for name in skills:
        assert (SRC / "mason" / "skills" / name / "SKILL.md").is_file()
