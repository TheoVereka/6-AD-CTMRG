#!/bin/bash

# Resume unfinished, not-yet-resumed iPEPS jobs started on/after 2026-09-03.
# Run this script from /scratch/izar/chye/0827core (or place it there first).

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CUTOFF="2026-09-03T00:00:00"
SPECIAL_D8_JOB="3143160"
NEAR_D9_SECONDS=$((3 * 86400 + 12 * 3600))  # 3.5 days
LONG_D9_SECONDS=$((6 * 86400))
MAIN="/home/chye/0801core/main_swave_LBFGS.py"
OUTPUT_ROOT="/scratch/izar/chye/0801core"
RUN_DIR="${SCRIPT_DIR}/resume_runs_$(date +%Y%m%d_%H%M%S)"

declare -a ids=()
declare -a states=()
declare -a elapsed_secs=()
declare -a j2s=()
declare -a ds=()
declare -a checkpoints=()
declare -a preferred=()
declare -a assigned=()
declare -a run_files=()
declare -A seen=()
declare -A resumed_sources=()

find_job_log() {
    local job_id="$1"
    local matches=("${SCRIPT_DIR}"/job-*-"${job_id}".out)
    if ((${#matches[@]} == 0)); then
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

echo "Collecting output directories that have already been resumed ..."
resume_evidence=("${SCRIPT_DIR}"/job-*.out "${OUTPUT_ROOT}"/*/run.log)
for evidence_file in "${resume_evidence[@]}"; do
    while IFS= read -r resume_path; do
        [[ -n "${resume_path}" ]] || continue
        resumed_sources["${resume_path%/*}"]=1
    done < <(sed -nE \
        's/^[[:space:]]*Resumed from[[:space:]]+([^[:space:]]+).*/\1/p' \
        "${evidence_file}")
done

echo "Searching Slurm jobs started on/after ${CUTOFF} ..."

while IFS='|' read -r job_id state start elapsed job_name; do
    [[ "${job_id}" =~ ^[0-9]+$ ]] || continue
    [[ -z "${seen[${job_id}]+x}" ]] || continue
    seen["${job_id}"]=1

    state="${state%% *}"
    state="${state%%+*}"
    [[ "${start}" != "Unknown" && "${start}" != "N/A" ]] || continue
    [[ "${start}" < "${CUTOFF}" ]] && continue
    [[ "${elapsed}" =~ ^[0-9]+$ ]] || elapsed=0

    # Ended jobs are the ones killed by their time limit.  For running jobs,
    # consider 3143160 explicitly and all jobs older than 3.5 days; D=9 is
    # checked below after reading the run metadata.
    if [[ "${state}" == "TIMEOUT" ]]; then
        :
    elif [[ "${state}" == "RUNNING" ]] && \
         { [[ "${job_id}" == "${SPECIAL_D8_JOB}" ]] || ((elapsed >= NEAR_D9_SECONDS)); }; then
        :
    else
        continue
    fi

    if ! log_file="$(find_job_log "${job_id}")"; then
        echo "WARNING: job ${job_id}: no job-*-${job_id}.out; skipping" >&2
        continue
    fi

    output_dir="$(sed -nE 's/^[[:space:]]*Output dir[[:space:]]*:[[:space:]]*//p' "${log_file}" | tail -n 1)"
    j2="$(sed -nE 's/.*J1=[^[:space:]]+[[:space:]]+J2=([-+0-9.eE]+).*/\1/p' "${log_file}" | head -n 1)"
    D="$(sed -nE 's/.*D_bond sweep[[:space:]]*:[[:space:]]*\[([0-9]+)\].*/\1/p' "${log_file}" | head -n 1)"

    if [[ -z "${output_dir}" || -z "${j2}" || -z "${D}" ]]; then
        echo "WARNING: job ${job_id}: could not read output dir/J2/D from ${log_file}; skipping" >&2
        continue
    fi

    if [[ ${resumed_sources[$output_dir]+present} == "present" ]]; then
        echo "Skipping job ${job_id}: ${output_dir} has already been resumed"
        continue
    fi

    # The special D=8 job is always included.  Other running jobs are included
    # only when they are D=9 and have already run for at least 3.5 days.
    if [[ "${state}" == "RUNNING" && "${job_id}" != "${SPECIAL_D8_JOB}" ]]; then
        [[ "${D}" == "9" && ${elapsed} -ge ${NEAR_D9_SECONDS} ]] || continue
    fi

    checkpoint_matches=("${output_dir}"/sweep_D"${D}"_chi*_best.pt)
    if ((${#checkpoint_matches[@]} == 0)); then
        echo "WARNING: job ${job_id}: no D=${D} best checkpoint in ${output_dir}; skipping" >&2
        continue
    fi
    checkpoint="${checkpoint_matches[$((${#checkpoint_matches[@]} - 1))]}"

    i=${#ids[@]}
    ids[i]="${job_id}"
    states[i]="${state}"
    elapsed_secs[i]="${elapsed}"
    j2s[i]="${j2}"
    ds[i]="${D}"
    checkpoints[i]="${checkpoint}"

    # D=9 jobs that have only run 3--5 days prefer a fresh seven-day slot.
    # D=8/other D and D=9 jobs that reached six days prefer a three-day slot.
    if [[ "${D}" == "9" ]] && ((elapsed < LONG_D9_SECONDS)); then
        preferred[i]=7
    else
        preferred[i]=3
    fi
    assigned[i]="${preferred[i]}"
done < <(sacct -u "${USER}" -S 2026-09-03 -X -n -P \
              -o JobIDRaw,State,Start,ElapsedRaw,JobName%100)

n=${#ids[@]}
if ((n == 0)); then
    echo "No jobs need resuming."
    exit 0
fi

# Keep the two queue lengths equal (or one apart for an odd total) while
# changing as few preferred assignments as possible.
preferred_seven=0
for value in "${preferred[@]}"; do
    ((value == 7)) && ((preferred_seven += 1))
done

lower=$((n / 2))
upper=$(((n + 1) / 2))
if ((preferred_seven < lower)); then
    target_seven=${lower}
elif ((preferred_seven > upper)); then
    target_seven=${upper}
else
    target_seven=${preferred_seven}
fi

if ((preferred_seven > target_seven)); then
    # Move the longest-running short-D9 candidates to the three-day group.
    flips=$((preferred_seven - target_seven))
    while ((flips > 0)); do
        pick=-1
        pick_elapsed=-1
        for ((i = 0; i < n; i++)); do
            if [[ "${assigned[i]}" == "7" ]] && ((elapsed_secs[i] > pick_elapsed)); then
                pick=${i}
                pick_elapsed=${elapsed_secs[i]}
            fi
        done
        assigned[pick]=3
        ((flips -= 1))
    done
elif ((preferred_seven < target_seven)); then
    # If seven-day jobs are short, move the least-long D=9 jobs first; only
    # use D=8/other D after all such D=9 choices are exhausted.
    flips=$((target_seven - preferred_seven))
    while ((flips > 0)); do
        pick=-1
        pick_elapsed=9223372036854775807
        for ((i = 0; i < n; i++)); do
            if [[ "${assigned[i]}" == "3" && "${ds[i]}" == "9" ]] && \
               ((elapsed_secs[i] < pick_elapsed)); then
                pick=${i}
                pick_elapsed=${elapsed_secs[i]}
            fi
        done
        if ((pick < 0)); then
            for ((i = 0; i < n; i++)); do
                if [[ "${assigned[i]}" == "3" ]] && ((elapsed_secs[i] < pick_elapsed)); then
                    pick=${i}
                    pick_elapsed=${elapsed_secs[i]}
                fi
            done
        fi
        assigned[pick]=7
        ((flips -= 1))
    done
fi

mkdir -p "${RUN_DIR}"

three_count=0
seven_count=0
for ((i = 0; i < n; i++)); do
    job_id="${ids[i]}"
    days="${assigned[i]}"
    run_file="${RUN_DIR}/resume_${job_id}_${days}days.run"
    run_files[i]="${run_file}"

    if [[ "${days}" == "7" ]]; then
        qos_line="#SBATCH --qos long"
        time_limit="167:59:50"
        ((seven_count += 1))
    else
        qos_line="##SBATCH --qos long"
        time_limit="71:59:50"
        ((three_count += 1))
    fi

    cat > "${run_file}" <<EOF
#!/bin/bash
${qos_line}
#SBATCH --partition gpu
#SBATCH --job-name=resume_${job_id}
#SBATCH -e ${SCRIPT_DIR}/job-%N-%j.error
#SBATCH -o ${SCRIPT_DIR}/job-%N-%j.out
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=${time_limit}
#SBATCH --exclude=i39

echo "***** START OF JOB ***** "
module load gcc/11.3.0
module load cuda/11.8.0
echo STARTING AT \`date\`

source /home/chye/venvs/6adctmrg_Izar/bin/activate
python ${MAIN} --J2 ${j2s[i]} --Ds ${ds[i]} --ansatz neel_legacy --resume ${checkpoints[i]}

echo FINISHED at \`date\`
echo " ***** END OF JOB ***** "
EOF
    chmod +x "${run_file}"
done

echo "Found ${n} jobs: ${three_count} x 3-day and ${seven_count} x 7-day."
printf '%-10s %-8s %-4s %-8s %-8s %s\n' "OLD_JOB" "STATE" "D" "J2" "DAYS" "CHECKPOINT"
for ((i = 0; i < n; i++)); do
    printf '%-10s %-8s %-4s %-8s %-8s %s\n' \
        "${ids[i]}" "${states[i]}" "${ds[i]}" "${j2s[i]}" "${assigned[i]}" "${checkpoints[i]}"
done

# Submit every resume first.  Record only running originals whose replacement
# was accepted by sbatch; cancellation happens in one final phase below.
declare -a cancel_ids=()
submission_failures=0
for ((i = 0; i < n; i++)); do
    echo "Submitting replacement for ${ids[i]} (${assigned[i]} days) ..."
    if submission_output="$(sbatch "${run_files[i]}" 2>&1)"; then
        echo "  ${submission_output}"
        if [[ "${states[i]}" == "RUNNING" ]]; then
            cancel_ids+=("${ids[i]}")
        fi
    else
        echo "  ERROR: ${submission_output}" >&2
        ((submission_failures += 1))
    fi
done

if ((${#cancel_ids[@]} > 0)); then
    echo "All submissions attempted; cancelling replaced running jobs: ${cancel_ids[*]}"
    scancel "${cancel_ids[@]}"
fi

if ((submission_failures > 0)); then
    echo "Finished with ${submission_failures} failed submission(s); their original running jobs were not cancelled." >&2
    exit 1
fi

echo "All resume jobs submitted successfully."
