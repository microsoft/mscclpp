# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
import sys
import importlib.util
from pathlib import Path

# Add the python package to sys.path so Sphinx can find it
project_root = Path(__file__).parent.parent
python_path = project_root / "python"
sys.path.insert(0, str(python_path))

# -- Project version -----------------------------------------------------
spec = importlib.util.spec_from_file_location("_version", python_path / "mscclpp" / "_version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
version = version_module.__version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mscclpp"
copyright = "2025, MSCCL++ Team"
author = "MSCCL++ Team"
release = "v" + version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_multiversion",
]

smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = None
smv_released_pattern = r"^tags/.*$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
# only mock the C-extension when using the source tree
autodoc_mock_imports = ["mscclpp._version", "mscclpp._mscclpp", "blake3", "cupy", "mpi4py", "numpy", "sortedcontainers"]
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
html_js_files = [
    "version-selector.js",
]


# Custom setup function to copy version-selector.js to root build directory
def setup(app):
    import shutil
    from pathlib import Path

    def copy_version_selector(app, exception):
        if exception is None:  # Only copy if build succeeded
            source = Path(app.srcdir) / "_static" / "version-selector.js"
            # Copy to root build directory
            dest_root = Path(app.outdir).parent / "_static"
            dest_root.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copy2(source, dest_root / "version-selector.js")

    app.connect("build-finished", copy_version_selector)
