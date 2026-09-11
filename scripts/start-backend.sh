#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
. scripts/env.sh

if [ "$#" -eq 0 ]; then
    set -- examples/customers.csv -p none
fi

exec uv run sift run "$@"
