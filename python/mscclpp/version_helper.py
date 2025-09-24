import subprocess
import os
import re

def get_git_revision():
    """Get the current git commit hash"""
    try:
        # Get the short commit hash - use the repository root
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        # Check if there are uncommitted changes
        try:
            subprocess.check_output(
                ['git', 'diff-index', '--quiet', 'HEAD'],
                cwd=repo_root
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
            
        if dirty:
            commit += '-dirty'
            
        return commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'

def get_git_branch():
    """Get the current git branch"""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'

def get_git_remote_url():
    """Get the git remote URL"""
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        url = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            cwd=repo_root,
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        # Remove any credentials from the URL for security
        url = re.sub(r'://[^@]+@', '://', url)
        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'

def get_package_version():
    """Extract version from setup.py or use default, and append git commit"""
    base_version = "0.7.0"  # Base version
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        setup_py = os.path.join(repo_root, 'setup.py')
        with open(setup_py, 'r') as f:
            content = f.read()
            # Look for version in setup()
            match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                base_version = match.group(1)
    except:
        pass
    
    # Get git commit hash
    commit = get_git_revision()
    
    # Create version string with commit
    if commit != 'unknown':
        # For development versions, append the commit hash
        # Format: base_version+git.commit_hash
        version = f"{base_version}+git.{commit}"
    else:
        version = base_version
    
    return version, base_version, commit

def write_version_file(output_path=None):
    """Write version information to a file
    
    Args:
        output_path: Path where to write the version file. If None, uses default location.
    
    Returns:
        dict: Version information dictionary
    """
    if output_path is None:
        # Default to writing _version.py in the same directory as this module
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_version.py')
    
    version_with_commit, base_version, commit = get_package_version()
    branch = get_git_branch()
    remote_url = get_git_remote_url()
    
    version_content = f'''# Auto-generated version file - DO NOT EDIT
# Generated at build time from git repository information

__version__ = "{version_with_commit}"
__base_version__ = "{base_version}"
__git_commit__ = "{commit}"
__git_branch__ = "{branch}"
__git_remote__ = "{remote_url}"

def get_version_info():
    """Get complete version information as a dictionary"""
    return {{
        "version": __version__,
        "base_version": __base_version__,
        "commit": __git_commit__,
        "branch": __git_branch__,
        "remote": __git_remote__
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
        print(f"  Git Branch: {{info['branch']}}")
        print(f"  Git Remote: {{info['remote']}}")
    return info
'''
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(version_content)
    
    print(f"Generated version file at {output_path}")
    print(f"  Version: {version_with_commit}")
    print(f"  Base Version: {base_version}")
    print(f"  Commit: {commit}")
    print(f"  Branch: {branch}")
    print(f"  Git Remote: {remote_url}")
    
    return {
        "version": version_with_commit,
        "base_version": base_version,
        "commit": commit,
        "branch": branch,
        "remote": remote_url
    }

if __name__ == "__main__":
    # For testing: generate version file
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else None
    info = write_version_file(output)
    print(f"\nVersion info: {info}")