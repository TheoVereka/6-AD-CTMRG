#!/bin/bash
# Submit every available D>=3 checkpoint as one independent Slurm job.

set -euo pipefail
shopt -s nullglob

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
submitted=0
skipped=0

for checkpoint in "$BUNDLE_DIR"/checkpoints/J2_*/D_*/tensor_best.pt; do
  d_dir="$(basename "$(dirname "$checkpoint")")"
  D="${d_dir#D_}"
  if (( D < 3 )); then
    continue
  fi
  j2_dir="$(basename "$(dirname "$(dirname "$checkpoint")")")"
  j2_text="${j2_dir#J2_}"
  J2_VALUE="${j2_text//p/.}"
  OUTPUT_JSON="$BUNDLE_DIR/results_straight_rows_v3/$j2_dir/D_${D}.json"
  if python "$BUNDLE_DIR/compute_six_correlation_lengths.py" \
      --checkpoint "$checkpoint" \
      --J2 "$J2_VALUE" \
      --output "$OUTPUT_JSON" \
      --check-only; then
    echo "SKIP existing $OUTPUT_JSON"
    ((skipped += 1))
    continue
  fi
  mkdir -p "$(dirname "$OUTPUT_JSON")"
  export_spec="ALL,BUNDLE_DIR=$BUNDLE_DIR,CHECKPOINT=$checkpoint,J2_VALUE=$J2_VALUE,OUTPUT_JSON=$OUTPUT_JSON"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "WOULD SUBMIT J2=${J2_VALUE}, D=${D} -> ${OUTPUT_JSON}"
  else
    sbatch \
      --job-name="nxi_${j2_text}_D${D}" \
      --chdir="$BUNDLE_DIR" \
      --export="$export_spec" \
      "$BUNDLE_DIR/run_one.run"
  fi
  ((submitted += 1))
done

echo "Selected $submitted job(s); skipped $skipped completed case(s); dry_run=$DRY_RUN."
