#!/usr/bin/env bash
# Start the resonator fit web app. Extra args are passed through, e.g.
#   ./run.sh --db "/path/to/other/Database" --port 8060 --host 0.0.0.0
cd "$(dirname "$0")" && exec python app.py "$@"
