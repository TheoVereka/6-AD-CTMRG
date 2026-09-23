#!/usr/bin/env bash
# Cancel only this workflow's jobs and remove only this workflow's raw results.
set -euo pipefail

if [[ "${1:-}" != "--yes" || $# -ne 1 ]]; then
    echo "Usage: bash clear_j2_continuation_izar.sh --yes" >&2
    exit 2
fi

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED="$(realpath -m "${HOME}/VBCJ2SeedContinuationIzar")"
if [[ "${BUNDLE_DIR}" != "${EXPECTED}" ]]; then
    echo "Refusing cleanup outside the expected Izar bundle." >&2
    echo "Resolved: ${BUNDLE_DIR}" >&2
    echo "Expected: ${EXPECTED}" >&2
    exit 10
fi

workflow_job_ids() {
    squeue --noheader --user "${USER}" --format '%A %j' 2>/dev/null |
    awk '$2 ~ /^c[0-9][0-9][0-9][lr][12]$/ ||
         $2 ~ /^vjc[0-9][0-9][0-9][lr][12]s[0-9][0-9]$/ {print $1}'
}

mapfile -t JOB_IDS < <(workflow_job_ids)
if (( ${#JOB_IDS[@]} > 0 )); then
    echo "Cancelling ${#JOB_IDS[@]} old/new J2-continuation jobs: ${JOB_IDS[*]}"
    scancel "${JOB_IDS[@]}"
else
    echo "No matching J2-continuation jobs are queued."
fi

# scancel is asynchronous. Do not remove a live process's output tree.
for _attempt in $(seq 1 30); do
    mapfile -t REMAINING < <(workflow_job_ids)
    (( ${#REMAINING[@]} == 0 )) && break
    sleep 2
done
mapfile -t REMAINING < <(workflow_job_ids)
if (( ${#REMAINING[@]} > 0 )); then
    echo "Refusing to delete results while jobs remain: ${REMAINING[*]}" >&2
    exit 12
fi

RESULTS="$(realpath -m "${BUNDLE_DIR}/Results_Izar_J2_sequences")"
LOGS="$(realpath -m "${BUNDLE_DIR}/slurm_logs")"
[[ "${RESULTS}" == "${BUNDLE_DIR}/Results_Izar_J2_sequences" ]] || exit 11
[[ "${LOGS}" == "${BUNDLE_DIR}/slurm_logs" ]] || exit 11
rm -rf -- "${RESULTS}" "${LOGS}"
mkdir -p "${LOGS}"
find "${BUNDLE_DIR}" -maxdepth 1 -type f \
    \( -name '*.out' -o -name '*.error' \) -delete
rm -f -- "${BUNDLE_DIR}/run_j2_sequence_izar.sh"
echo "Removed workflow results and logs; seeds/scripts were preserved."
