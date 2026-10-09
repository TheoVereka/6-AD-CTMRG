#!/usr/bin/env bash
# Submit the explicit 30-job QOS allocation in manifest priority order.
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"
DRY_RUN=0
REFRESH_QOS=0
for ARG in "$@"; do
    case "${ARG}" in
        --dry-run) DRY_RUN=1 ;;
        --refresh-qos) REFRESH_QOS=1 ;;
        *) echo "Usage: bash submit_all.sh [--dry-run] [--refresh-qos]" >&2; exit 2 ;;
    esac
done
if [[ "${REFRESH_QOS}" == "1" ]]; then
    python3 prepare_jobs.py --auto
fi
if ! grep -q '"ready": true' qos_allocation.json; then
    echo "QOS allocation pending: use --refresh-qos on Izar, or supply both actual running counts to prepare_jobs.py." >&2
    exit 2
fi
[[ -f run_pair.sh ]] || { echo "Missing run_pair.sh" >&2; exit 2; }
for NAME in renyi2_twoc3.py renyi2_spectral.py correlation_length.py core_C3.py; do
    [[ -f "code/${NAME}" ]] || { echo "Missing code/${NAME}" >&2; exit 2; }
done

# Validate the whole plan before submitting any job.
COUNT=0
while IFS=, read -r PRIORITY J2 D CHI PAIR NORMAL_ENV SWAPPED_ENV QOS WALLTIME JOB_NAME JOB OUTPUT; do
    [[ "${PRIORITY}" == "priority" ]] && continue
    COUNT=$((COUNT + 1))
    TAG="${J2/./p}"
    [[ "${PRIORITY}" == "${COUNT}" && "${D}" == "8" && "${CHI}" == "80" ]] || {
        echo "Malformed or out-of-order manifest row ${COUNT}" >&2; exit 2;
    }
    [[ -f "seeds/J2_${TAG}/D_8/tensor_best.pt" && -f "${JOB}" ]] || {
        echo "Missing seed or job for J2=${J2}, pair=${PAIR}" >&2; exit 2;
    }
    grep -Fxq "#SBATCH --qos=${QOS}" "${JOB}" && \
    grep -Fxq "#SBATCH --time=${WALLTIME}" "${JOB}" || {
        echo "Header/manifest mismatch: ${JOB}" >&2; exit 2;
    }
done < job_manifest.csv
[[ "${COUNT}" == "30" ]] || { echo "Expected 30 allocated jobs, found ${COUNT}" >&2; exit 2; }
if [[ "${DRY_RUN}" == "0" ]]; then
    command -v sbatch >/dev/null 2>&1 || { echo "sbatch unavailable; use --dry-run locally." >&2; exit 2; }
    mkdir -p slurm_logs
fi
while IFS=, read -r PRIORITY J2 D CHI PAIR NORMAL_ENV SWAPPED_ENV QOS WALLTIME JOB_NAME JOB OUTPUT; do
    [[ "${PRIORITY}" == "priority" ]] && continue
    printf '%02d  J2=%-5s D=%s chi=%s pair=%s qos=%-6s time=%s  %s\n' \
        "${PRIORITY}" "${J2}" "${D}" "${CHI}" "${PAIR}" "${QOS}" "${WALLTIME}" "${JOB}"
    if [[ "${DRY_RUN}" == "0" ]]; then
        sbatch --chdir="${BUNDLE_DIR}" --export="ALL,BUNDLE_DIR=${BUNDLE_DIR}" "${BUNDLE_DIR}/${JOB}"
    fi
done < job_manifest.csv
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run complete: ${COUNT} jobs; nothing submitted."
else
    echo "Submitted ${COUNT} separate jobs."
fi
