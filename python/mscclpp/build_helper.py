"""Build helper utilities for MSCCLPP package setup"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

# Import version_helper carefully to avoid circular imports
try:
    from . import version_helper
except ImportError:
    # If relative import fails (e.g., when called from setup.py), try direct import
    import version_helper


class CustomBuildPy(build_py):
    """Custom build_py command to generate version file before building"""
    
    def run(self):
        """Run the build_py command with version file generation"""
        # Generate version file before building
        try:
            # Get the python/mscclpp directory
            if hasattr(self, 'build_lib'):
                # During actual build, use build_lib path
                mscclpp_build_dir = os.path.join(self.build_lib, 'mscclpp')
                os.makedirs(mscclpp_build_dir, exist_ok=True)
                version_file = os.path.join(mscclpp_build_dir, '_version.py')
            else:
                # Fallback to source directory
                mscclpp_dir = os.path.dirname(os.path.abspath(__file__))
                version_file = os.path.join(mscclpp_dir, '_version.py')
            
            # Also generate in source directory for development
            src_version_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '_version.py'
            )
            
            version_helper.write_version_file(src_version_file)
            print(f"Successfully generated version file at {src_version_file}")
            
            # Copy to build directory if different
            if hasattr(self, 'build_lib') and version_file != src_version_file:
                import shutil
                shutil.copy2(src_version_file, version_file)
                print(f"Copied version file to {version_file}")
            
        except Exception as e:
            print(f"Warning: Failed to generate version file: {e}")
            # Create a fallback version file
            self._create_fallback_version_file()
        
        # Continue with normal build
        super().run()
    
    def _create_fallback_version_file(self):
        """Create a fallback version file if git info is not available"""
        # Create in both source and build directories
        for base_dir in [os.path.dirname(os.path.abspath(__file__)),
                        getattr(self, 'build_lib', None)]:
            if base_dir is None:
                continue
                
            if hasattr(self, 'build_lib') and base_dir == self.build_lib:
                version_file = os.path.join(base_dir, 'mscclpp', '_version.py')
                os.makedirs(os.path.dirname(version_file), exist_ok=True)
            else:
                version_file = os.path.join(base_dir, '_version.py')
            
            with open(version_file, 'w') as f:
                f.write('''# Auto-generated fallback version file
__version__ = "0.1.0"
__git_commit__ = "unknown"
__git_branch__ = "unknown"
__git_remote__ = "unknown"

def get_version_info():
    return {
        "version": __version__,
        "commit": __git_commit__,
        "branch": __git_branch__,
        "remote": __git_remote__
    }

def show_version(verbose=True):
    info = get_version_info()
    if verbose:
        print("MSCCLPP Version Information:")
        print(f"  Package Version: {info['version']}")
        print(f"  Git Commit: {info['commit']} (build info unavailable)")
    return info
''')
            print(f"Created fallback version file at {version_file}")


class CMakeExtension(Extension):
    """CMake-based extension"""
    
    def __init__(self, name: str, sourcedir: str = "") -> None:
        """Initialize CMake extension
        
        Args:
            name: Extension name
            sourcedir: Source directory for CMake project
        """
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """CMake build extension command"""
    
    def build_extension(self, ext: CMakeExtension) -> None:
        """Build a CMake extension
        
        Args:
            ext: The CMakeExtension to build
        """
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Determine build type
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Configure CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        
        build_args = []

        # Add any user-provided CMake arguments
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Set parallel build level if not already set
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        # Create build directory
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Configure the project
        print(f"Configuring CMake project with args: {cmake_args}")
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        
        # Build the project
        print(f"Building CMake project with args: {build_args}")
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


def get_package_version():
    """Get the package version from various sources
    
    Returns:
        str: The package version
    """
    # Default version
    version = "0.1.0"
    
    # Try to get version from environment variable
    if "MSCCLPP_VERSION" in os.environ:
        return os.environ["MSCCLPP_VERSION"]
    
    # Try to get version from VERSION file
    try:
        version_file = Path(__file__).parent.parent.parent / "VERSION"
        if version_file.exists():
            version = version_file.read_text().strip()
    except:
        pass
    
    return version


def get_long_description():
    """Get the long description from README
    
    Returns:
        str: The long description
    """
    try:
        readme_file = Path(__file__).parent.parent.parent / "README.md"
        if readme_file.exists():
            return readme_file.read_text()
    except:
        pass
    
    return "Microsoft Collective Communication Library++ (MSCCL++) Python bindings"


# Export the build commands and utilities
__all__ = [
    'CustomBuildPy',
    'CMakeExtension', 
    'CMakeBuild',
    'get_package_version',
    'get_long_description'
]