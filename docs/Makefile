# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXMULTIVERSION ?= sphinx-multiversion
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Usage:"
	@echo "  make html          - Build single-version HTML (fast, for development)"
	@echo "  make multiversion  - Build all versions with sphinx-multiversion"
	@echo "  make clean         - Remove build directory"
	@echo ""
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile generate-versions multiversion clean

# Generate versions.js from git tags before building
generate-versions:
	@python3 generate_versions.py

# Build all documentation versions using sphinx-multiversion
# Use this for production builds or to test version switching
multiversion: generate-versions
	@cd .. && python3 -m setuptools_scm --force-write-version-files
	@$(SPHINXMULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Clean build directory
clean:
	@rm -rf $(BUILDDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# This builds single-version only (fast for development).
%: Makefile generate-versions
	@cd .. && python3 -m setuptools_scm --force-write-version-files
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
