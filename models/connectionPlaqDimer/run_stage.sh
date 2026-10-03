#!/usr/bin/env bash
# Run one node submitted by submit_all.sh; all paths are inside this bundle.
set -euo pipefail

: "${D:?D is required}"
: "${CONNECTION:?CONNECTION is required}"
: "${NODE:?NODE is required}"
: "${T:?T is required}"
: "${PREV_NODE:?PREV_NODE is required}"

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

[[ "${D}" == 6 || "${D}" == 7 ]] || { echo "Invalid D=${D}" >&2; exit 2; }
[[ "${CONNECTION}" == plaq || "${CONNECTION}" == dimer ]] || {
    echo "Invalid CONNECTION=${CONNECTION}" >&2; exit 2;
}
[[ "${T}" =~ ^[0-8]$ ]] || { echo "Invalid T=${T}" >&2; exit 2; }

case "${NODE}" in
    A)
        MAIN=main_C3.py
        [[ "${T}" == 0 && "${PREV_NODE}" == none ]] || {
            echo "A requires T=0 and no predecessor" >&2; exit 2;
        }
        ;;
    B|C|D|E|F|G|H|I)
        MAIN=main_C3_LBFGS.py
        [[ "${PREV_NODE}" != none ]] || { echo "Missing predecessor" >&2; exit 2; }
        ;;
    O|P|Q|R|S|T|U)
        MAIN=main_C3.py
        [[ "${PREV_NODE}" == A ]] || { echo "Branch requires A" >&2; exit 2; }
        ;;
    *) echo "Invalid NODE=${NODE}" >&2; exit 2 ;;
esac

if [[ "${D}" == 6 ]]; then HOURS=70
else HOURS=166; fi

TREE_DIR="${BUNDLE_DIR}/results/D${D}_${CONNECTION}"
OUTPUT_DIR="${TREE_DIR}/${NODE}"
mkdir -p "${OUTPUT_DIR}"

select_best_checkpoint() {
    python - "$1" "$2" <<'PY'
from pathlib import Path
import re
import sys

directory = Path(sys.argv[1])
bond_dimension = sys.argv[2]
pattern = re.compile(rf"sweep_D{re.escape(bond_dimension)}_chi([0-9]+)_best\.pt")
candidates = []
for path in directory.glob(f"sweep_D{bond_dimension}_chi*_best.pt"):
    match = pattern.fullmatch(path.name)
    if match and path.stat().st_size > 0:
        candidates.append((int(match.group(1)), path))
if not candidates:
    sys.exit(f"No best checkpoint in {directory} for D={bond_dimension}")
print(max(candidates)[1])
PY
}

ARGS=(
    --connection "${CONNECTION}"
    --t "${T}"
    --ansatz twoc3
    --Ds "${D}"
    --hours "${HOURS}"
    --gpu
    --ngpu 1
    --output-dir "${OUTPUT_DIR}"
)

if [[ "${NODE}" == A ]]; then
    ARGS+=(--mean-field-init)
else
    PREV_DIR="${TREE_DIR}/${PREV_NODE}"
    [[ -s "${PREV_DIR}/COMPLETED.stage" ]] || {
        echo "Predecessor has no completion marker: ${PREV_DIR}" >&2; exit 3;
    }
    RESUME_CKPT="$(select_best_checkpoint "${PREV_DIR}" "${D}")"
    ARGS+=(--no-mean-field-init --resume "${RESUME_CKPT}" --resume-tensors-only)
fi
if [[ "${MAIN}" == main_C3_LBFGS.py ]]; then
    ARGS+=(--optimizer lbfgs)
fi

echo "Node=${NODE}; D=${D}; connection=${CONNECTION}; t=${T}; predecessor=${PREV_NODE}"
echo "Main=${MAIN}; hours=${HOURS}; output=${OUTPUT_DIR}"
if [[ "${NODE}" != A ]]; then echo "Resume=${RESUME_CKPT}"; fi
python -u "${BUNDLE_DIR}/${MAIN}" "${ARGS[@]}"

[[ -s "${OUTPUT_DIR}/sweep_results.json" ]] || {
    echo "Run returned without sweep_results.json: ${OUTPUT_DIR}" >&2; exit 4;
}
BEST_CKPT="$(select_best_checkpoint "${OUTPUT_DIR}" "${D}")"
printf 'node=%s\nD=%s\nconnection=%s\nt=%s\ncheckpoint=%s\ncompleted_utc=%s\n' \
    "${NODE}" "${D}" "${CONNECTION}" "${T}" "${BEST_CKPT}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT_DIR}/COMPLETED.stage.tmp"
mv -f "${OUTPUT_DIR}/COMPLETED.stage.tmp" "${OUTPUT_DIR}/COMPLETED.stage"
echo "COMPLETED ${NODE}: ${BEST_CKPT}"
