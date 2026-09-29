#!/usr/bin/env bash
# Build a download-safe snapshot of partial/completed LastIzar results and logs.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

ARCHIVE="${1:-LastIzar_results_snapshot.tar.gz}"
DIAG_DIR="download_diagnostics"
mkdir -p "${DIAG_DIR}"

squeue -u "${USER}" -o '%i|%j|%T|%M|%R' \
    > "${DIAG_DIR}/squeue_LastIzar.psv" || true
sacct -u "${USER}" -S 2026-09-28 \
    --format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList -P \
    > "${DIAG_DIR}/sacct_LastIzar.psv" || true

: > "${DIAG_DIR}/scontrol_dependency_jobs.txt"
while IFS='|' read -r JOB_ID JOB_NAME JOB_STATE JOB_TIME JOB_REASON; do
    [[ "${JOB_NAME}" =~ ^L[1-5] ]] || continue
    if [[ "${JOB_REASON}" == *Dependency* ]]; then
        scontrol show job -o "${JOB_ID}" \
            >> "${DIAG_DIR}/scontrol_dependency_jobs.txt" || true
    fi
done < "${DIAG_DIR}/squeue_LastIzar.psv"

ITEMS=(download_diagnostics)
[[ -d Results_LastIzar ]] && ITEMS+=(Results_LastIzar)
[[ -d slurm_logs ]] && ITEMS+=(slurm_logs)

TMP_ARCHIVE="${ARCHIVE}.tmp.$$"
set +e
tar --warning=no-file-changed --ignore-failed-read -czf "${TMP_ARCHIVE}" \
    "${ITEMS[@]}"
TAR_STATUS=$?
set -e
if (( TAR_STATUS > 1 )); then
    echo "tar failed with status ${TAR_STATUS}" >&2
    exit "${TAR_STATUS}"
fi
mv -f "${TMP_ARCHIVE}" "${ARCHIVE}"
echo "Snapshot created: ${BUNDLE_DIR}/${ARCHIVE}"
du -h "${ARCHIVE}"
