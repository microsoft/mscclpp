#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrapper around sphinx-multiversion that patches copy_tree to generate
_version.py in each tag checkout. This is needed because setuptools_scm
generates _version.py at build time, but sphinx-multiversion uses
`git archive` which only contains committed files.

Usage (called by Makefile):
    python3 build_multiversion.py <sourcedir> <outputdir> [sphinx-opts...]
"""

import os
import re
import subprocess
import sys

import sphinx_multiversion.git as smv_git
from sphinx_multiversion import main as smv_main

# Save the original copy_tree
_original_copy_tree = smv_git.copy_tree


def _patched_copy_tree(gitroot, src, dst, reference, sourcepath="."):
    """Call original copy_tree, then generate _version.py from the VERSION file."""
    _original_copy_tree(gitroot, src, dst, reference, sourcepath)

    # Extract version from the tag name (e.g., "v0.9.0" -> "0.9.0")
    refname = getattr(reference, "refname", "") or ""
    match = re.search(r"v(\d+\.\d+\.\d+)", refname)
    if not match:
        return

    version = match.group(1)
    version_py_dir = os.path.join(dst, "python", "mscclpp")
    if os.path.isdir(version_py_dir):
        version_py = os.path.join(version_py_dir, "_version.py")
        if not os.path.exists(version_py):
            with open(version_py, "w") as f:
                f.write(f'__version__ = "{version}"\n')


# Monkey-patch
smv_git.copy_tree = _patched_copy_tree

if __name__ == "__main__":
    sys.exit(smv_main(sys.argv[1:]))
