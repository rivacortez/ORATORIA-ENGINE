#!/usr/bin/env bash
# Install the local git hooks.
#
# Opt-in rather than automatic. A repository that silently installs hooks on
# clone is doing something to a developer's machine that they did not ask for,
# and the first time one blocks a legitimate push nobody knows where it came
# from.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

install -m 755 scripts/pre-push .git/hooks/pre-push

echo "installed: .git/hooks/pre-push"
echo
echo "This is a stand-in for server-side branch protection, which needs GitHub"
echo "Pro on a private repository. It runs on one machine and can be skipped"
echo "with --no-verify. CI remains the real gate."
