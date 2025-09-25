from __future__ import annotations
import os
import subprocess
from pathlib import Path
from setuptools import setup
from setuptools_scm import ScmVersion

def custom_version_scheme(version: ScmVersion) -> str:
    """
    Custom version scheme that generates format: v0.7.0.dev<num_commit>+g<sha>
    And generates _version.py file with all version information
    """
    # Get the base version
    if version.exact:
        base_version = version.tag.lstrip('v') if version.tag else "0.7.0"
        version_string = f"v{base_version}"
    else:
        if version.tag:
            base_version = version.tag.lstrip('v')
            version_string = f"v{base_version}.dev{version.distance}"
        else:
            base_version = "0.7.0"
            version_string = f"v{base_version}.dev{version.distance or 0}"
    
    # Get short commit hash with 'g' prefix
    commit_hash = version.node[:7] if version.node else "unknown"
    version_string += f"+g{commit_hash}"
    
    # Add .dirty suffix if uncommitted changes
    if version.dirty:
        version_string += ".dirty"
    
    # Generate _version.py file
    _generate_version_file(version_string, version)
    
    return version_string

def _generate_version_file(version_string: str, version: ScmVersion):
    """Generate the _version.py file with all version information"""
    try:
        # Get additional git information
        git_branch = _get_git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        git_remote = _get_git_output(["git", "config", "--get", "remote.origin.url"])
        full_commit = _get_git_output(["git", "rev-parse", "HEAD"])
        
        # Parse version components
        if version.exact:
            parsed_base_version = version.tag.lstrip('v') if version.tag else "0.8.0"
            git_distance = 0
        else:
            parsed_base_version = version.tag.lstrip('v') if version.tag else "0.8.0"
            git_distance = version.distance or 0
        
        # Create version file content
        version_content = f'''# Auto-generated version file - DO NOT EDIT
# Generated at build time from git repository information

__version__ = "{version_string}"
__base_version__ = "{parsed_base_version}"
__git_commit__ = "{version.node[:7] if version.node else 'unknown'}"
__git_commit_full__ = "{full_commit}"
__git_branch__ = "{git_branch}"
__git_remote__ = "{git_remote}"
__git_dirty__ = {version.dirty}
__git_distance__ = {git_distance}
__scm_version__ = "{version_string}"

def get_version_info():
    """Get complete version information as a dictionary"""
    return {{
        "version": __version__,
        "base_version": __base_version__,
        "commit": __git_commit__,
        "commit_full": __git_commit_full__,
        "branch": __git_branch__,
        "remote": __git_remote__,
        "dirty": __git_dirty__,
        "distance": __git_distance__,
        "scm_version": __scm_version__
    }}

def show_version(verbose=True):
    """Display version information
    
    Args:
        verbose (bool): If True, print to stdout. Always returns the version dict.
    
    Returns:
        dict: Version information dictionary
    """
    info = get_version_info()
    if verbose:
        print("MSCCLPP Version Information:")
        print(f"  Package Version: {{info['version']}}")
        print(f"  Base Version: {{info['base_version']}}")
        print(f"  Git Commit: {{info['commit']}}")
        print(f"  Full Git Commit: {{info['commit_full']}}")
        print(f"  Git Branch: {{info['branch']}}")
        print(f"  Git Remote: {{info['remote']}}")
        print(f"  Working Tree Dirty: {{info['dirty']}}")
        print(f"  Distance from Tag: {{info['distance']}}")
    return info
'''
        
        # Write to version file
        version_file = Path("python/mscclpp/_version.py")
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(version_content)
        
        print(f"Generated version file: {version_file}")
        print(f"Version: {version_string}")
        
    except Exception as e:
        print(f"Warning: Could not generate version file: {e}")

def _get_git_output(cmd: list) -> str:
    """Helper to get git command output"""
    try:
        return subprocess.check_output(
            cmd, 
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

setup(
    use_scm_version={
        "version_scheme": custom_version_scheme,
    }
)