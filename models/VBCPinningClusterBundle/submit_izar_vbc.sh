#!/bin/bash
# Submit replica 1 through the 0827-style three-day resume job and replica 2
# through the seven-day resume job. Unsafe 31-GiB (D,chi) pairs are rejected.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

for REQUIRED in main_C3.py core_C3.py run_vbc_branch.sh izar_memory_guard.sh seed_orientations.csv 3daysResumingjob.run 7daysResumingjob.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || { echo "Missing ${REQUIRED}" >&2; exit 1; }
done

read -r -a J2_VALUES <<< "${J2_VALUES_TEXT:-0.29 0.30 0.31 0.32}"
read -r -a D_VALUES <<< "${D_VALUES_TEXT:-8 9}"
BRANCHES=(plaquette dimer-plaquette)

# Empirically safe with lookahead on the 31.74-GiB Izar GPUs.
izar_chi_for_D() {
    case "$1" in
        8) echo 104 ;;
        9) echo 108 ;;
        *) return 1 ;;
    esac
}

seed_for() {
    local J2="$1"
    local D="$2"
    local TAG="${J2/./p}"
    while [[ "${TAG}" == *0 ]]; do TAG="${TAG%0}"; done
    echo "${BUNDLE_DIR}/seeds/J2_${TAG}/D_${D}/tensor_best.pt"
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
blocked=0
for J2 in "${J2_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        if CHI="$(izar_chi_for_D "${D}")"; then
            :
        elif [[ "${ALLOW_UNSAFE_IZAR:-0}" == "1" && -n "${CHI_OVERRIDE:-}" ]]; then
            CHI="${CHI_OVERRIDE}"
            echo "WARNING: forcing unvalidated Izar pair D=${D}, chi=${CHI}" >&2
        else
            echo "BLOCKED for 31-GiB Izar: D=${D}. D9 chi=126 is already known to OOM; D10/D11 high-chi runs belong on Kuma." >&2
            blocked=$((blocked + 1))
            continue
        fi

        SEED_CKPT="$(seed_for "${J2}" "${D}")"
        if [[ ! -f "${SEED_CKPT}" ]]; then
            echo "Missing bundled replica-1 seed: ${SEED_CKPT}" >&2
            exit 3
        fi

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
                SHORT_BRANCH=p
                [[ "${BRANCH}" == "dimer-plaquette" ]] && SHORT_BRANCH=d
                COMMON="ALL,BUNDLE_DIR=${BUNDLE_DIR},D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION}"

                sbatch --chdir="${BUNDLE_DIR}" \
                    --job-name="i${J2/./}D${D}${SHORT_BRANCH}r1" \
                    --export="${COMMON},REPLICA=1,SEED_CKPT=${SEED_CKPT}" \
                    "${BUNDLE_DIR}/3daysResumingjob.run"
                submitted=$((submitted + 1))

                sbatch --chdir="${BUNDLE_DIR}" \
                    --job-name="i${J2/./}D${D}${SHORT_BRANCH}r2" \
                    --export="${COMMON},REPLICA=2" \
                    "${BUNDLE_DIR}/7daysResumingjob.run"
                submitted=$((submitted + 1))
            done
        done
    done
done

echo "Izar: submitted=${submitted}, blocked_D_values=${blocked}."
