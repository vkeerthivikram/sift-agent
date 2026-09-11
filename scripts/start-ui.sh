#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
. scripts/env.sh

exec uv run sift-ui "$@"
