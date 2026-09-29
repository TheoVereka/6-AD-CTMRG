#!/usr/bin/env bash
# Remove only the superseded LastIzar h=.005 task families and their blocked
# h=0 dependants.  The existing task1 D=6 adiabatic data are untouched.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${BUNDLE_DIR}/Results_LastIzar"
LOG_ROOT="${BUNDLE_DIR}/slurm_logs"

cancelled=0
while IFS='|' read -r job_id job_name reason; do
    [[ -n "${job_id}" ]] || continue
    if [[ "${reason}" == *DependencyNeverSatisfied* \
            && "${job_name}" =~ ^L[2345] ]]; then
        echo "scancel ${job_id} (${job_name}: ${reason})"
        scancel "${job_id}"
        cancelled=$((cancelled + 1))
    fi
done < <(squeue -h -u "${USER}" -t PD -o '%A|%j|%R')

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
resolved_results="$(cd -- "${RESULTS_ROOT}" && pwd)"
[[ "${resolved_results}" == "${BUNDLE_DIR}/Results_LastIzar" ]] || {
    echo "Refusing unsafe result root: ${resolved_results}" >&2
    exit 70
}

for task in \
        task2_D6_adam_pin task3_D6_lbfgs_pin \
        task4_D5_adam_pin task5_D5_lbfgs_pin; do
    target="${resolved_results}/${task}"
    [[ "${target}" == "${resolved_results}/"* ]] || {
        echo "Refusing unsafe deletion target: ${target}" >&2
        exit 70
    }
    if [[ -e "${target}" ]]; then
        echo "Deleting obsolete ${target}"
        rm -rf -- "${target}"
    fi
done

# These logs belong only to the removed L2/L3/L4/L5 task families.
find "${LOG_ROOT}" -maxdepth 1 -type f \
    \( -name 'L2*' -o -name 'L3*' -o -name 'L4*' -o -name 'L5*' \) \
    -print -delete

# scp merges directories and therefore does not remove retired static files
# already present on the cluster.  Remove the old launchers only; task1 result
# data remain untouched and are now represented as insurance 1 by the plotter.
find "${BUNDLE_DIR}/jobs" -maxdepth 1 -type f \
    \( -name 't1_d6_*.run' -o -name 't2_*.run' -o -name 't3_*.run' \
       -o -name 't4_*.run' -o -name 't5_*.run' \) \
    -print -delete

echo "Cleanup complete: cancelled ${cancelled} blocked jobs; task1 was preserved."
