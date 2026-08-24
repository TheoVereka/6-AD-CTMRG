#!/bin/bash
# Submit every missing legacy-Neel ordinary result. D>=10 is split three ways.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_correlation_lengths.sh" "$@"
