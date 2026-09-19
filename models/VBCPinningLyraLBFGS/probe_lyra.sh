#!/usr/bin/env bash
# Read-only Lyra capability probe plus two short default-QOS allocations.
# Run on lyra.hpc.epfl.ch and send the generated report back before setup.
set -uo pipefail

REPORT="${1:-lyra_probe_$(date +%Y%m%d_%H%M%S).txt}"
exec > >(tee "${REPORT}") 2>&1

section() {
    printf '\n===== %s =====\n' "$1"
}

section "identity and login message"
date --iso-8601=seconds
hostname -f || hostname
id
printf 'HOME=%s\n' "${HOME:-UNSET}"
printf 'SCRATCH=%s\n' "${SCRATCH:-UNSET}"
printf 'WORK=%s\n' "${WORK:-UNSET}"
printf 'SHELL=%s\n' "${SHELL:-UNSET}"
if [[ -r /etc/motd ]]; then
    sed -n '1,160p' /etc/motd
fi

section "Slurm version and user associations"
sinfo --version || true
sacctmgr -n -P show assoc where user="${USER}" \
    format=Cluster,Account,User,Partition,QOS,DefaultQOS 2>&1 || true

section "Lyra QOS"
sacctmgr -n -P show qos \
    format=Name,Priority,MaxWall,MaxTRESPJ,MaxJobsPU,MaxSubmitPU 2>&1 \
    | grep -E '^(normal|long|build|debug)\|' || true

section "partitions and node resources"
sinfo -p b200,rtx6000 -o '%P|%a|%l|%D|%c|%m|%G|%f' 2>&1 || true
scontrol show partition b200 2>&1 || true
scontrol show partition rtx6000 2>&1 || true

section "frontend module candidates"
type module 2>&1 || true
module purge 2>&1 || true
module -t avail 2>&1 \
    | grep -Ei '(^|/)(gcc|cuda|nvhpc|python|py-virtualenv|py-pip)(/|$)' \
    | sort -u || true
for package in gcc cuda nvhpc python py-virtualenv; do
    printf '\n--- module spider %s ---\n' "${package}"
    module spider "${package}" 2>&1 || true
done

gpu_probe='set -uo pipefail
echo "node=$(hostname -f || hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader \
    || nvidia-smi
echo "--- module candidates on compute node ---"
module purge 2>&1 || true
module -t avail 2>&1 | grep -Ei "(^|/)(gcc|cuda|nvhpc|python|py-virtualenv|py-pip)(/|$)" | sort -u || true
echo "--- system Python ---"
command -v python3 || true
python3 --version 2>&1 || true'

section "B200 default-QOS node probe"
srun --partition=b200 --gpus=1 --ntasks=1 \
    --cpus-per-task=12 --time=00:05:00 bash -lc "${gpu_probe}" || true

section "RTX6000 default-QOS node probe"
srun --partition=rtx6000 --gpus=1 --ntasks=1 \
    --cpus-per-task=16 --time=00:05:00 bash -lc "${gpu_probe}" || true

section "report"
printf 'Saved report: %s\n' "$(pwd)/${REPORT}"
