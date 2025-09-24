"""Setup script for MSCCLPP Python package"""

import sys
import os
import subprocess
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

# Add python/mscclpp to path to import version_helper
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python', 'mscclpp'))

try:
    import version_helper
except ImportError:
    version_helper = None


class GenerateVersionFile:
    """Mixin to generate version file"""
    
    def generate_version_file(self):
        """Generate the _version.py file with git information"""
        if version_helper is None:
            print("Warning: version_helper not available, creating fallback version file")
            self.create_fallback_version_file()
            return
            
        try:
            # Determine where to write the version file
            if hasattr(self, 'build_lib'):
                # During build, write to build directory
                version_file = os.path.join(self.build_lib, 'mscclpp', '_version.py')
                os.makedirs(os.path.dirname(version_file), exist_ok=True)
            else:
                # During egg_info or sdist, write to source directory
                version_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'python', 'mscclpp', '_version.py'
                )
                
            print(f"Generating version file at {version_file}")
            version_helper.write_version_file(version_file)
            
            # Also ensure it's in the source directory for development
            src_version_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'python', 'mscclpp', '_version.py'
            )
            if version_file != src_version_file:
                import shutil
                shutil.copy2(version_file, src_version_file)
                print(f"Also copied to source at {src_version_file}")
                
        except Exception as e:
            print(f"Warning: Failed to generate version file: {e}")
            self.create_fallback_version_file()
    
    def create_fallback_version_file(self):
        """Create a fallback version file if git info is not available"""
        fallback_content = '''# Auto-generated fallback version file
__version__ = "0.7.0"
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
        print(f"  Git Commit: {info['commit']}")
        print(f"  Git Branch: {info['branch']}")
        print(f"  Git Remote: {info['remote']}")
    return info
'''
        
        # Write to both source and build directories if available
        for base_dir in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python', 'mscclpp'),
            getattr(self, 'build_lib', None)
        ]:
            if base_dir is None:
                continue
                
            if hasattr(self, 'build_lib') and base_dir == self.build_lib:
                version_file = os.path.join(base_dir, 'mscclpp', '_version.py')
                os.makedirs(os.path.dirname(version_file), exist_ok=True)
            else:
                version_file = os.path.join(base_dir, '_version.py')
                
            with open(version_file, 'w') as f:
                f.write(fallback_content)
            print(f"Created fallback version file at {version_file}")


class CustomBuildPy(build_py, GenerateVersionFile):
    """Custom build_py command to generate version file before building"""
    
    def run(self):
        """Run the build_py command with version file generation"""
        self.generate_version_file()
        super().run()


class CustomEggInfo(egg_info, GenerateVersionFile):
    """Custom egg_info command to generate version file"""
    
    def run(self):
        """Run the egg_info command with version file generation"""
        self.generate_version_file()
        super().run()


class CustomSdist(sdist, GenerateVersionFile):
    """Custom sdist command to include version file in source distribution"""
    
    def run(self):
        """Run the sdist command with version file generation"""
        self.generate_version_file()
        super().run()


# Get package metadata
VERSION = "0.7.0"
LONG_DESCRIPTION = """Microsoft Collective Communication Library++ (MSCCL++) Python bindings"""

try:
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        LONG_DESCRIPTION = readme_file.read_text()
except:
    pass

# Package configuration
setup(
    name="mscclpp",
    version=VERSION,
    author="Microsoft",
    author_email="",
    description="MSCCLPP Python bindings with version tracking",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/mscclpp",
    
    # Custom build commands
    cmdclass={
        "build_py": CustomBuildPy,
        "egg_info": CustomEggInfo,
        "sdist": CustomSdist,
    },
    
    # Package configuration
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    
    # Package data
    package_data={
        'mscclpp': ['*.py', '*.pyi', '_version.py'],
    },
    include_package_data=True,
    
    # Installation requirements
    python_requires=">=3.8",
    install_requires=[
        # Add any runtime dependencies here
    ],
    
    # Development dependencies (optional)
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
        ],
        'test': [
            'pytest',
            'numpy',
        ],
    },
    
    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'mscclpp-version=mscclpp.__main__:main',
        ],
    },
    
    # Additional metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    
    # Ensure package is not zipped
    zip_safe=False,
)