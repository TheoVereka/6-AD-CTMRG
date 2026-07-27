#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_FILE="${SCRIPT_DIR}/test_correlation_length.run"

for D_BOND in 2 3 4 5 6; do
    sbatch \
        --job-name="corr-length-D${D_BOND}" \
        "${RUN_FILE}" \
        "${D_BOND}"
done
