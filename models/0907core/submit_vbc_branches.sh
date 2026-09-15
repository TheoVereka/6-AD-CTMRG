#!/bin/bash
# Submit the decisive same-manifold VBC comparison.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/singleFileSbatchVBC.run"

if [[ ! -f "${SCRIPT_DIR}/main_C3.py" || ! -f "${SCRIPT_DIR}/core_C3.py" || ! -f "${SBATCH_FILE}" ]]; then
    echo "Copy main_C3.py, core_C3.py, and both VBC scripts into one cluster directory." >&2
    exit 1
fi

# Default pilot: the most informative central VBC points and two D values.
# For the publication run, override, for example:
#   J2_VALUES_TEXT="0.28 0.30 0.32 0.34" D_VALUES_TEXT="8 9 10 11" ./submit_vbc_branches.sh
read -r -a J2_VALUES <<< "${J2_VALUES_TEXT:-0.30 0.32}"
read -r -a D_VALUES <<< "${D_VALUES_TEXT:-8 10}"
read -r -a ORIENTATIONS <<< "${ORIENTATIONS_TEXT:-0}"
read -r -a REPLICAS <<< "${REPLICAS_TEXT:-1 2}"
BRANCHES=(plaquette dimer-plaquette)

chi_for_D() {
    case "$1" in
        8)  echo 160 ;;
        9)  echo 180 ;;
        10) echo 180 ;;
        11) echo 160 ;;
        12) echo 192 ;;
        *)
            echo "No tested chi default for D=$1; export CHI_OVERRIDE." >&2
            return 1
            ;;
    esac
}

submitted=0
failures=0
for J2 in "${J2_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        CHI="${CHI_OVERRIDE:-$(chi_for_D "${D}")}"
        for ORIENTATION in "${ORIENTATIONS[@]}"; do
            for BRANCH in "${BRANCHES[@]}"; do
                for REPLICA in "${REPLICAS[@]}"; do
                    SHORT_BRANCH="p"
                    [[ "${BRANCH}" == "dimer-plaquette" ]] && SHORT_BRANCH="d"
                    JOB_NAME="v${J2/./}D${D}${SHORT_BRANCH}o${ORIENTATION}r${REPLICA}"
                    SEED_EXPORT=""
                    # Replica 1 reuses the old state; replica 2 remains a fresh
                    # deterministic start, so the replica spread is meaningful.
                    if [[ -n "${SEED_ROOT:-}" && "${REPLICA}" == "1" ]]; then
                        J2_TAG="${J2/./p}"
                        while [[ "${J2_TAG}" == *0 ]]; do J2_TAG="${J2_TAG%0}"; done
                        CANDIDATE="${SEED_ROOT}/J2_${J2_TAG}/2tensor_twoC3/D_${D}/tensor_best.pt"
                        if [[ -f "${CANDIDATE}" ]]; then
                            SEED_EXPORT=",SEED_CKPT=${CANDIDATE}"
                        else
                            echo "No old tensor at ${CANDIDATE}; using seeded random init." >&2
                        fi
                    fi
                    if sbatch \
                        --chdir="${SCRIPT_DIR}" \
                        --job-name="${JOB_NAME}" \
                        --export="ALL,D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION},REPLICA=${REPLICA}${SEED_EXPORT}" \
                        "${SBATCH_FILE}"; then
                        submitted=$((submitted + 1))
                    else
                        failures=$((failures + 1))
                    fi
                done
            done
        done
    done
done

echo "Submitted ${submitted} branch-following jobs; failures=${failures}."
(( failures == 0 ))
