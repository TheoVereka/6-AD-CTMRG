#!/bin/bash
# Kuma: one seeded replica, J2=0.30, D=5,...,11, and three pinning sources.
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

for REQUIRED in main_C3.py core_C3.py run_vbc_branch.sh seed_orientations.csv kumaVBC.run; do
    [[ -f "${BUNDLE_DIR}/${REQUIRED}" ]] || { echo "Missing ${REQUIRED}" >&2; exit 1; }
done

J2="0.30"
D_VALUES=(5 6 7 8 9 10 11)
BRANCHES=(plaquette dimer-plaquette rank-split)
OUTROOT="${OUTROOT:-${BUNDLE_DIR}/Results_VBC_three}"

kuma_chi_for_D() {
    case "$1" in
        5) echo 50 ;;
        6) echo 72 ;;
        7) echo 91 ;;
        8) echo 104 ;;
        9) echo 180 ;;
        10) echo 180 ;;
        11) echo 160 ;;
        *) return 1 ;;
    esac
}

seed_for() {
    local D="$1"
    echo "${BUNDLE_DIR}/seeds/J2_0p3/D_${D}/tensor_best.pt"
}

# Return geometrical group indices as rank1:rank2:rank3, from the strongest
# (most-negative) NN correlation to the weakest.
rank_order_for() {
    local D="$1"
    awk -F, -v d="${D}" '
        NR > 1 && $1 == "0p3" && $2 == d {
            g[0]=$3+0; g[1]=$4+0; g[2]=$5+0
            imax=0; imin=0
            for (i=1; i<3; i++) {
                if (g[i] > g[imax]) imax=i
                if (g[i] < g[imin]) imin=i
            }
            imid=3-imax-imin
            printf "%d:%d:%d\n", imin, imid, imax
            found=1
            exit
        }
        END { if (!found) exit 1 }
    ' "${BUNDLE_DIR}/seed_orientations.csv"
}

submitted=0
for D in "${D_VALUES[@]}"; do
    CHI="${CHI_OVERRIDE:-$(kuma_chi_for_D "${D}")}"
    SEED_CKPT="$(seed_for "${D}")"
    [[ -f "${SEED_CKPT}" ]] || { echo "Missing ${SEED_CKPT}" >&2; exit 3; }

    if ! RANK_ORDER="$(rank_order_for "${D}")"; then
        echo "No rank order in seed manifest for J2=${J2}, D=${D}" >&2
        exit 4
    fi
    # Texture-aware singled group from the seed manifest: weak for a PVB-like
    # seed and strong for a dimer-like seed.
    ORIENTATION="$(awk -F, -v d="${D}" \
        'NR > 1 && $1 == "0p3" && $2 == d { print $9; exit }' \
        "${BUNDLE_DIR}/seed_orientations.csv")"
    [[ "${ORIENTATION}" =~ ^[012]$ ]] || { echo "Missing orientation for D=${D}" >&2; exit 4; }

    for BRANCH in "${BRANCHES[@]}"; do
        case "${BRANCH}" in
            plaquette) SHORT_BRANCH=p ;;
            dimer-plaquette) SHORT_BRANCH=d ;;
            rank-split) SHORT_BRANCH=r ;;
        esac
        sbatch --chdir="${BUNDLE_DIR}" \
            --job-name="k3D${D}${SHORT_BRANCH}" \
            --export="ALL,BUNDLE_DIR=${BUNDLE_DIR},OUTROOT=${OUTROOT},D=${D},CHI=${CHI},J2=${J2},BRANCH=${BRANCH},ORIENTATION=${ORIENTATION},RANK_ORDER=${RANK_ORDER},REPLICA=1,SEED_CKPT=${SEED_CKPT}" \
            "${BUNDLE_DIR}/kumaVBC.run"
        submitted=$((submitted + 1))
    done
done

echo "Kuma three-source sweep: submitted=${submitted} (replica 1 only)."
