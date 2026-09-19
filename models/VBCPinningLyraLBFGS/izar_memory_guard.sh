#!/usr/bin/env bash
# Izar's validated 31.74-GiB configurations, including chi+D lookahead.
set -euo pipefail
: "${D:?D must be exported}"
: "${CHI:?CHI must be exported}"

case "${D}:${CHI}" in
    6:72|7:91|8:104|9:108) ;;
    *)
        echo "Refusing unvalidated Izar pair D=${D}, chi=${CHI}." >&2
        echo "Allowed: D6/chi72, D7/chi91, D8/chi104, D9/chi108." >&2
        exit 70
        ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
    TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "Izar memory guard: D=${D}, chi=${CHI}, GPU memory=${TOTAL_MIB:-unknown} MiB"
fi
