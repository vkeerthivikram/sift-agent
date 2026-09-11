#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
. scripts/env.sh

scripts/start-backend.sh "$@"
exec scripts/start-ui.sh
