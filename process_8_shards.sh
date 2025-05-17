#!/bin/bash

# Check if language parameter is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <language_code> [starting_shard]"
    echo "Supported language codes: en, fi, fr, sv"
    exit 1
fi

LANG=$1
START_SHARD=${2:-1}  # Default to 1 if not provided

# Calculate the end shard (process 8 shards starting from START_SHARD)
END_SHARD=$((START_SHARD + 7))
if [ $END_SHARD -gt 32 ]; then
    END_SHARD=32
fi

NUM_SHARDS=$((END_SHARD - START_SHARD + 1))

echo "Processing shards $START_SHARD to $END_SHARD for language $LANG"

# Create a temporary job script
JOB_SCRIPT="batch_${LANG}_${START_SHARD}_${END_SHARD}.sh"
LOGS_DIR="slurm-logs"
mkdir -p $LOGS_DIR

cat > $JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=trankit_${LANG}
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=${LOGS_DIR}/%A_%a.out
#SBATCH --error=${LOGS_DIR}/%A_%a.err
#SBATCH --account=project_462000353
#SBATCH --partition=small-g
#SBATCH --array=0-$((NUM_SHARDS-1))

# Load environment
source venv/bin/activate

# Load PyTorch module
module use /appl/local/csc/modulefiles
module load pytorch/2.0

# Calculate which shard to process based on array task ID
SHARD_NUM=\$((${START_SHARD} + SLURM_ARRAY_TASK_ID))
echo "Processing ${LANG} shard \${SHARD_NUM}"

# Run the Python script
python parse_shard.py ${LANG} \${SHARD_NUM}

echo "Completed processing ${LANG} shard \${SHARD_NUM}"
EOL

# Make the script executable
chmod +x $JOB_SCRIPT

# Submit the job
echo "Submitting job array to process shards $START_SHARD-$END_SHARD for $LANG"
sbatch $JOB_SCRIPT

echo "Job submitted. Check logs directory for progress."