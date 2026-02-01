#!/bin/sh
# Install git hooks for this repository

HOOKS_DIR="$(dirname "$0")"
GIT_HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"

echo "Git hooks installed successfully"
