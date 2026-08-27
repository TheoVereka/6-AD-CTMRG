#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

write_run() {
    local j2="$1"
    local D="$2"
    local token="${j2#0.}"
    local filename="${token}nelD${D}.run"
    local qos_line
    local time_limit

    if [[ "${D}" -eq 9 ]]; then
        qos_line="#SBATCH --qos long"
        time_limit="167:59:50"
    else
        qos_line="##SBATCH --qos long"
        time_limit="71:59:50"
    fi

    cat > "${filename}" <<EOF
#!/bin/bash
${qos_line}
#SBATCH --partition gpu
#SBATCH -e ./job-%N-%j.error
#SBATCH -o ./job-%N-%j.out
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=40G                   # Job memory request
#SBATCH --time=${time_limit}             # Time limit
#SBATCH --exclude=i39

echo "***** START OF JOB ***** "
module load gcc/11.3.0
module load cuda/11.8.0
echo STARTING AT \`date\`

source /home/chye/venvs/6adctmrg_Izar/bin/activate
python /home/chye/0801core/main_swave.py --J2 ${j2} --Ds ${D}  --ansatz neel_legacy

echo FINISHED at \`date\`
echo " ***** END OF JOB ***** "
#cp -r ./  /home/chye/0801core/
EOF
    chmod +x "${filename}"
    echo "Submitting J2=${j2} D=${D}: ${filename}"
    sbatch "${filename}"
}

# Submit every D=9 job first, in the requested J2 order.
for j2 in 0 0.2 0.21 0.22 0.23 0.235 0.24 0.245 0.25 0.255 0.26 0.265 0.27 0.275 0.28; do
    write_run "${j2}" 9
done

# Then submit D<=8 from the smallest D to the largest D.
for j2 in 0.245; do
    write_run "${j2}" 4
done

for j2 in 0.235 0.245; do
    write_run "${j2}" 5
done

for j2 in 0.22 0.23 0.235 0.245 0.25 0.265; do
    write_run "${j2}" 6
done

for j2 in 0.2 0.22 0.23 0.235 0.245 0.265 0.27 0.275 0.28; do
    write_run "${j2}" 7
done

for j2 in 0.22 0.23 0.235 0.245 0.26 0.265 0.27 0.275 0.28; do
    write_run "${j2}" 8
done
