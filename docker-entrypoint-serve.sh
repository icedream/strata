#!/bin/sh
# docker-entrypoint-serve.sh - start serve/server.py straight from a finished engine config.
#
# Upstream's docker/entrypoint.sh (0.2.x) plans a model, tunes it and starts it through
# `strata launch`; it lists CONFIG and HOST as legacy and ignores them (line 68: "$legacy is
# from the older image and is ignored here"). This container names one finished config instead:
# the pack, the locally tuned hipBLASLt table, the sampling block and the elastic cache are all
# in that JSON, so nothing is planned, offered or measured at start.
#
# Required:  CONFIG=/opt/strata/config/<name>.json
# Optional:  PORT (8080), HOST (0.0.0.0), STRATA_API_KEY (required below), extra args pass through.
set -eu

: "${CONFIG:?CONFIG must name the engine config to start}"
[ -f "$CONFIG" ] || { echo "CONFIG=$CONFIG does not exist in the container." >&2; exit 1; }

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# serve/server.py takes --api-key from STRATA_API_KEY. 0.2.x made the key optional (the server
# starts and warns); the port is published on the host here, so an empty key is a mistake, not a
# choice - refuse it rather than serve the model and its tools to the LAN.
if [ -z "${STRATA_API_KEY:-}" ]; then
  echo "STRATA_API_KEY is empty: refusing to publish :$PORT without one." >&2
  exit 1
fi

cd "$(dirname "$0")"
echo "Starting serve/server.py: config $CONFIG, ${HOST}:${PORT}"
exec .venv/bin/python serve/server.py --engine strata --config "$CONFIG" \
     --host "$HOST" --port "$PORT" "$@"
