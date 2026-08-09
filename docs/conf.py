"""Sphinx configuration for the pyodsp documentation.

The API pages are generated from the docstrings in the source, so most of
the reference material is written once, next to the code it describes.
The narrative pages under docs/guide are written by hand.
"""

import os
import sys
from pathlib import Path

# Documenting an uninstalled checkout should work, so the package root goes
# on the path rather than relying on `pip install -e .` having been run.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# -- Project ---------------------------------------------------------------

project = "pyodsp"
author = "Hideaki Nakao"
copyright = "2026, Hideaki Nakao"
# Sphinx convention: `release` is the full version, `version` the X.Y series.
release = "0.2.0"
version = "0.2"

# -- General ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Every page is Markdown; these are the MyST features the guide uses.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# -- Autodoc ---------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
# The docstrings here explain what a class is for in its class docstring and
# document __init__'s parameters there too, so the two are merged.
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True

# mpi4py is optional at runtime and painful to install on a docs builder, but
# pyodsp.dec.*.run_mpi imports it at module level.
autodoc_mock_imports = ["mpi4py"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = False
# Most of the result and scenario types are dataclasses documented with an
# `Attributes:` section. Without this, napoleon emits a py:attribute for each
# one *and* autodoc documents the field itself, so every attribute is listed —
# and warned about — twice. As an ivar field list they are rendered once.
napoleon_use_ivar = True

# -- Intersphinx -----------------------------------------------------------

# Pyomo objects appear all over these signatures, so linking them is worth
# the fetch. A builder with no network just drops the links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pyomo": ("https://pyomo.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_timeout = 10
# Offline builds should warn about unreachable inventories, not fail.
intersphinx_disabled_reftypes = ["*.std:doc"]

nitpicky = False
# `make strict` builds with -n, where these two are noise rather than signal:
# a TypeVar has nothing to link to, and Sphinx splits the annotation of an
# annotated module-level dict on its comma before trying to resolve it.
nitpick_ignore = [
    ("py:obj", "pyodsp.alg.bm.cuts.T"),
    ("py:class", "Dict[str"),
]

# -- HTML ------------------------------------------------------------------

html_theme = "furo"
html_title = f"pyodsp {release}"
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []
html_theme_options = {
    "source_repository": "https://github.com/hideakinakao/pyodsp",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
html_copy_source = False

# ReadTheDocs sets this; it switches the canonical URL and the version menu.
if os.environ.get("READTHEDOCS") == "True":
    html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
