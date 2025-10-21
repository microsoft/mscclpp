# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mscclpp"
copyright = "2025, MSCCL++ Team"
author = "MSCCL++ Team"
release = "v0.8.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys

sys.path.insert(0, os.path.abspath("../python"))

extensions = [
    "breathe",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
# only mock the C-extension when using the source tree
autodoc_mock_imports = ["mscclpp._mscclpp", "cupy", "mpi4py", "numpy", "sortedcontainers"]
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Breathe configuration
breathe_projects = {"mscclpp": "./doxygen/xml"}
breathe_default_project = "mscclpp"

# Mermaid configuration
mermaid_version = "11.0.0"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
