"""Root conftest: every test — including ``src/`` doctests — runs config-blind.

Configuration discovery reads the environment, ``~/.config``, and the
*current directory* (``./slab.toml``). Without isolation here, a developer
who follows the HPC tutorial and runs ``slab config init`` in this checkout
would change what the test suite resolves — workspace roots, pseudo roots,
the ``qe`` command. This conftest sits at the repository root so it covers
``tests/`` and the doctests collected from ``src/slab/`` alike (a fixture in
``tests/conftest.py`` reaches only the former).
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Neutralize all three config layers plus the workspace override.

    Tests that exercise configuration set these up themselves (writing a
    ``slab.toml`` into ``tmp_path`` and chdir-ing there, or pointing
    ``$SLAB_CONFIG`` at a file).
    """
    monkeypatch.delenv("SLAB_CONFIG", raising=False)
    monkeypatch.delenv("SLAB_SITE_CONFIG", raising=False)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    # The machine-memory store reads this one; a developer with memories of
    # their own must not have them enter the suite's catalogs.
    monkeypatch.delenv("SLAB_MEMORY_DIR", raising=False)
    # Runs read this one to stamp the session that created them. 'mason chat'
    # and 'mason run' export it for their child processes, and that export
    # outlives the CLI call inside one pytest process, so clear it for every
    # test; a test that wants a session stamp sets the variable itself.
    monkeypatch.delenv("SLAB_SESSION", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config-isolated"))
    # Rootstock discovers installs on its own: $ROOTSTOCK_ROOT, then a user
    # config at a literal ~/.config/rootstock/config.toml that ignores
    # $XDG_CONFIG_HOME. On a machine with a real install configured (an HPC
    # login node), tests asserting the *unconfigured* behavior would see it.
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    try:
        import rootstock.config as _rootstock_config
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            _rootstock_config,
            "DEFAULT_CONFIG_FILE",
            tmp_path / "rootstock-config-isolated" / "config.toml",
        )
    # The project layer is discovered from the working directory, so the
    # default cwd must be a directory with no slab.toml in it.
    neutral = tmp_path / "cwd-isolated"
    neutral.mkdir(exist_ok=True)
    monkeypatch.chdir(neutral)
