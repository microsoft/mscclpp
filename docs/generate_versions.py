# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Generate versions.js from git tags for the documentation version selector.

This script reads git tags matching the sphinx-multiversion pattern (vX.Y.Z)
and generates a JavaScript file containing the version list. This ensures the
version selector stays in sync with available documentation versions without
requiring manual updates.

Usage:
    python generate_versions.py

The script should be run before building documentation with sphinx-multiversion.
"""

import json
import re
import subprocess
from pathlib import Path


def get_git_tags():
    """Get all version tags from git matching vX.Y.Z pattern.

    Filters out versions before v0.4.0 as they don't have compatible docs structure
    for sphinx-multiversion.
    """
    try:
        result = subprocess.run(
            ["git", "tag", "-l", "v*.*.*"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = result.stdout.strip().split("\n")
        # Filter to match sphinx-multiversion pattern: ^v\d+\.\d+\.\d+$
        version_pattern = re.compile(r"^v\d+\.\d+\.\d+$")
        valid_tags = []
        for tag in tags:
            if tag and version_pattern.match(tag):
                # Filter out versions before v0.4.0 (no compatible docs structure)
                match = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag)
                if match:
                    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    # Include v0.4.0 and later
                    if major > 0 or (major == 0 and minor >= 4):
                        valid_tags.append(tag)
        return valid_tags
    except subprocess.CalledProcessError:
        return []


def version_sort_key(version):
    """Extract (major, minor, patch) tuple for sorting."""
    match = re.match(r"v(\d+)\.(\d+)\.(\d+)", version)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return (0, 0, 0)


def generate_versions_js(output_path):
    """Generate versions.js file from git tags."""
    tags = get_git_tags()

    # Sort versions in descending order (newest first)
    tags.sort(key=version_sort_key, reverse=True)

    # Build the version list with main (dev) first
    version_list = [{"name": "main (dev)", "path": "", "version": "main"}]

    for i, version in enumerate(tags):
        name = f"{version} (latest)" if i == 0 else version
        version_list.append({"name": name, "path": version, "version": version})

    # Generate JavaScript content
    js_content = f"""\
// Auto-generated from git tags by generate_versions.py - do not edit manually
// Run 'python generate_versions.py' or 'make html' to regenerate
const DEFINED_VERSIONS = {json.dumps(version_list, indent=4)};
"""

    # Write to output path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(js_content)
    print(f"Generated {output_path} with {len(version_list)} versions")


if __name__ == "__main__":
    # Generate versions.js in _static directory
    script_dir = Path(__file__).parent
    output_file = script_dir / "_static" / "versions.js"
    generate_versions_js(output_file)
