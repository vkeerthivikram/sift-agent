if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi
