#!/usr/bin/env bash

PROJECT_ROOT=$(dirname "$(realpath "$0")")/..
LINT_CPP=false
LINT_PYTHON=false
DRY_RUN=false

usage() {
    echo "Usage: $0 [cpp] [py] [dry]"
    echo "  cpp     Lint C++ code"
    echo "  py      Lint Python code"
    echo "  dry     Dry run mode (no changes made)"
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        cpp)
            LINT_CPP=true
            ;;
        py)
            LINT_PYTHON=true
            ;;
        dry)
            DRY_RUN=true
            ;;
        *)
            echo "Error: Unknown argument '$arg'"
            usage
            exit 1
            ;;
    esac
done

# If no cpp or py specified, default to both
if [ "$LINT_CPP" = false ] && [ "$LINT_PYTHON" = false ]; then
    LINT_CPP=true
    LINT_PYTHON=true
fi

if $LINT_CPP; then
    echo "Linting C++ code..."
    # Find all git-tracked files with .c/.h/.cpp/.hpp/.cc/.cu/.cuh extensions
    files=$(git -C "$PROJECT_ROOT" ls-files --cached | grep -E '\.(c|h|cpp|hpp|cc|cu|cuh)$' | sed "s|^|$PROJECT_ROOT/|")
    if [ -n "$files" ]; then
        if $DRY_RUN; then
            clang-format -style=file --dry-run $files
        else
            clang-format -style=file -i $files
        fi
    fi
fi

if $LINT_PYTHON; then
    echo "Linting Python code..."
    # Find all git-tracked files with .py extension
    files=$(git -C "$PROJECT_ROOT" ls-files --cached | grep -E '\.py$' | sed "s|^|$PROJECT_ROOT/|")
    if [ -n "$files" ]; then
        if $DRY_RUN; then
            python3 -m black --check --diff $files
        else
            python3 -m black $files
        fi
    fi
fi
