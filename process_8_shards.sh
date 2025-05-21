#!/bin/bash
# Check if language parameter is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <language_code> [starting_shard]"
    echo "Supported language codes: en, fi, fr, sv"
    exit 1
fi
LANG=$1
START_SHARD=${2:-1} # Default to 1 if not provided
# Calculate the end shard (process 8 shards starting from START_SHARD)
END_SHARD=$((START_SHARD + 7))
if [ $END_SHARD -gt 32 ]; then
    END_SHARD=32
fi
echo "Processing shards $START_SHARD to $END_SHARD for language $LANG"
# Create a temporary job script
JOB_SCRIPT="batch_${LANG}_${START_SHARD}_${END_SHARD}.sh"
LOGS_DIR="slurm-logs"
mkdir -p $LOGS_DIR
cat > $JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=trankit_${LANG}
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:mi250:8
#SBATCH --mem=128G
#SBATCH --time=32:00:00
#SBATCH --output=${LOGS_DIR}/%j.out
#SBATCH --error=${LOGS_DIR}/%j.err
#SBATCH --account=project_462000353
#SBATCH --partition=small-g

# Load environment
source venv/bin/activate

# Load PyTorch module
module use /appl/local/csc/modulefiles
module load pytorch/2.0

echo "Starting to process shards ${START_SHARD}-${END_SHARD} for language ${LANG}"

# Process shards in parallel, one per GPU
for i in \$(seq 0 7); do
    SHARD_NUM=\$((${START_SHARD} + i))
    if [ \$SHARD_NUM -le ${END_SHARD} ]; then
        echo "Starting process for ${LANG} shard \${SHARD_NUM}"
        srun --ntasks=1 --gres=gpu:mi250:1 --mem=16G \
            python parse_shard.py ${LANG} \${SHARD_NUM} > ${LOGS_DIR}/${LANG}_shard_\${SHARD_NUM}.log 2>&1 &
    fi
done

# Wait for all background processes to complete
wait
echo "Completed processing all shards for ${LANG}"
EOL

# Make the script executable
chmod +x $JOB_SCRIPT

# Submit the job
echo "Submitting job to process shards $START_SHARD-$END_SHARD for $LANG"
sbatch $JOB_SCRIPT
echo "Job submitted. Check logs directory for progress."