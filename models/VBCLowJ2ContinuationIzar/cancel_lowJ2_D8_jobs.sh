#!/usr/bin/env bash
# Cancel only D=8 jobs submitted by this bundle.  No other Izar jobs match the
# exact low-J2 job-name grammar below.
set -euo pipefail

mapfile -t job_ids < <(
    squeue --noheader --user="${USER}" --format='%A|%j' \
    | awk -F'|' '$2 ~ /^i8[dp](01|02|04|08)[mx][0-9]+$/ {print $1}' \
    | sort -u
)

if (( ${#job_ids[@]} == 0 )); then
    echo "No queued/running D=8 jobs from VBCLowJ2ContinuationIzar were found."
    exit 0
fi

printf 'Cancelling %d low-J2 Izar D=8 jobs:\n' "${#job_ids[@]}"
printf '  %s\n' "${job_ids[@]}"
scancel "${job_ids[@]}"
echo "Cancellation request sent."
