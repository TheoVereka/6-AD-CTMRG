#!/bin/bash
# Conservative whitelist for Izar's 31.74-GiB GPUs, including chi+D lookahead.
set -euo pipefail
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"

SAFE=0
[[ "${D}:${CHI}" == "8:104" ]] && SAFE=1
[[ "${D}:${CHI}" == "9:108" ]] && SAFE=1

if [[ "${SAFE}" != "1" && "${ALLOW_UNSAFE_IZAR:-0}" != "1" ]]; then
    echo "Refusing unvalidated Izar GPU-memory pair D=${D}, chi=${CHI}." >&2
    echo "Validated pairs: D=8 chi=104 (lookahead 112), D=9 chi=108 (lookahead 117)." >&2
    echo "D=9 chi=126 has documented OOM on a 31.74-GiB card." >&2
    echo "Use Kuma, or explicitly export ALLOW_UNSAFE_IZAR=1 after choosing a reduced chi." >&2
    exit 70
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "Izar memory guard: D=${D}, chi=${CHI}, GPU memory=${TOTAL_MIB:-unknown} MiB"
fi

