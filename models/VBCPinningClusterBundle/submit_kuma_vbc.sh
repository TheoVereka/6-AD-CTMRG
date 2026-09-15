#!/bin/bash
# High-chi Kuma counterpart using the same self-contained folder and seeds.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
read -r -a J2_VALUES <<< "${J2_VALUES_TEXT:-0.30 0.32}"
read -r -a D_VALUES <<< "${D_VALUES_TEXT:-9 10 11}"
read -r -a REPLICAS <<< "${REPLICAS_TEXT:-1 2}"
BRANCHES=(plaquette dimer-plaquette)

for REQUIRED in main_C3.py core_C3.py run_vbc_branch.sh seed_orientations.csv kumaVBC.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || { echo "Missing ${REQUIRED}" >&2; exit 1; }
done

kuma_chi_for_D() {
    case "$1" in
        8) echo 160 ;;
        9) echo 180 ;;
        10) echo 180 ;;
        11) echo 160 ;;
        12) echo 192 ;;
        *) return 1 ;;
    esac
}

orientation_for() {
    local J2="$1"
    local D="$2"
    local TAG="${J2/./p}"
    while [[ "${TAG}" == *0 ]]; do TAG="${TAG%0}"; done
    awk -F, -v tag="${TAG}" -v d="${D}" \
        'NR > 1 && $1 == tag && $2 == d { print $9; found=1; exit }
         END { if (!found) exit 1 }' \
        "${BUNDLE_DIR}/seed_orientations.csv"
}

submitted=0
for J2 in "${J2_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        CHI="${CHI_OVERRIDE:-$(kuma_chi_for_D "${D}")}"
        TAG="${J2/./p}"
        while [[ "${TAG}" == *0 ]]; do TAG="${TAG%0}"; done
        OLD_SEED="${BUNDLE_DIR}/seeds/J2_${TAG}/D_${D}/tensor_best.pt"
        if [[ -n "${ORIENTATIONS_TEXT:-}" ]]; then
            read -r -a ORIENTATIONS <<< "${ORIENTATIONS_TEXT}"
        else
            if ! AUTO_ORIENTATION="$(orientation_for "${J2}" "${D}")"; then
                echo "No seed-aware orientation for J2=${J2}, D=${D}" >&2
                exit 4
            fi
            ORIENTATIONS=("${AUTO_ORIENTATION}")
        fi
        for ORIENTATION in "${ORIENTATIONS[@]}"; do
            for BRANCH in "${BRANCHES[@]}"; do
                for REPLICA in "${REPLICAS[@]}"; do
                    SHORT_BRANCH=p
                    [[ "${BRANCH}" == "dimer-plaquette" ]] && SHORT_BRANCH=d
                    SEED_EXPORT=""
                    if [[ "${REPLICA}" == "1" ]]; then
                        [[ -f "${OLD_SEED}" ]] || { echo "Missing ${OLD_SEED}" >&2; exit 3; }
                        SEED_EXPORT=",SEED_CKPT=${OLD_SEED}"
                    fi
                    sbatch --chdir="${BUNDLE_DIR}" \
                        --job-name="k${J2/./}D${D}${SHORT_BRANCH}r${REPLICA}" \
                        --export="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION},REPLICA=${REPLICA}${SEED_EXPORT}" \
                        "${BUNDLE_DIR}/kumaVBC.run"
                    submitted=$((submitted + 1))
                done
            done
        done
    done
done

echo "Kuma: submitted=${submitted}."
