#!/usr/bin/env bash
# A running-safe snapshot: partial directories and logs are intentionally kept.
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
ARCHIVE="D7DimerJ2_0p26_snapshot.tar.gz"
mkdir -p download_diagnostics
squeue -u "${USER}" -o '%i|%j|%T|%M|%R' > download_diagnostics/squeue.psv || true
sacct -u "${USER}" -S 2026-09-29 \
    --format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList -P \
    > download_diagnostics/sacct.psv || true
items=(download_diagnostics)
[[ -d Results_D7DimerJ2_0p26 ]] && items+=(Results_D7DimerJ2_0p26)
[[ -d slurm_logs ]] && items+=(slurm_logs)
[[ -f submission_job_ids.tsv ]] && items+=(submission_job_ids.tsv)
tmp="${ARCHIVE}.tmp.$$"
set +e
tar --warning=no-file-changed --ignore-failed-read -czf "${tmp}" "${items[@]}"
status=$?
set -e
(( status <= 1 )) || { echo "tar failed: ${status}" >&2; exit "${status}"; }
mv -f "${tmp}" "${ARCHIVE}"
du -h "${ARCHIVE}"
