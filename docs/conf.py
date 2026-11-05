"""Sphinx configuration for cbc_spin_priors."""

from importlib.metadata import version as _version

project = "cbc_spin_priors"
author = "Colm Talbot"
copyright = "2026, Colm Talbot"
release = _version("cbc_spin_priors")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# the docstrings use numpy-style sections underlined with "="
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_typehints = "none"
autosummary_generate = False

exclude_patterns = ["_build"]

# fail on broken cross-references, except for the informal types in the
# docstrings and bilby classes, which have no inventory here
nitpicky = True
nitpick_ignore = [
    ("py:class", "optional"),
    ("py:class", "array_like"),
    ("py:class", "module"),
]
nitpick_ignore_regex = [
    ("py:.*", r"bilby\..*"),
]

html_theme = "furo"
html_title = "cbc_spin_priors"

mathjax3_config = {
    "tex": {
        "macros": {
            "amax": r"a_{\max}",
            "chieff": r"\chi_\mathrm{eff}",
        }
    }
}

latex_elements = {
    "preamble": r"\newcommand{\amax}{a_{\max}}\newcommand{\chieff}{\chi_\mathrm{eff}}",
}
