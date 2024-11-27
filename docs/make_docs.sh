#!/usr/bin/env bash

# Set the package name from the directory above this file (should be in ./docs/)
PACKAGE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"
PACKAGE_NAME=$(basename "$PACKAGE_DIR")

# Creates the documentation for the package
MODULES=("$PACKAGE_NAME" "examples" "tests")
# Create a string listing out all modules prepended with ./
MODULE_STRING=$(printf "./%s " "${MODULES[@]}")

# Create coverage badge
cd "$PACKAGE_DIR" && \
pip install pytest pytest-cov coverage genbadge[tests,coverage] -q --disable-pip-version-check && \
DEBIAN_FRONTEND=noninteractive pytest && \
genbadge tests -i junit.xml -o docs/tests.svg && \
genbadge coverage -i coverage.xml -o docs/coverage.svg && \

# We assume the appropriate virtual environment has already been activated
cd "$PACKAGE_DIR" && \
pip install pdoc -q --disable-pip-version-check && \
pdoc -t ./docs/theme -o ./docs $MODULE_STRING && \
echo "Successfully generated documentation at $PACKAGE_DIR/docs."
