set shell := ["sh", "-cu"]

default:
    @just --list

setup:
    uv sync --extra ui

backend *args:
    ./scripts/start-backend.sh {{args}}

ui *args:
    ./scripts/start-ui.sh {{args}}

all *args:
    ./scripts/start-all.sh {{args}}
