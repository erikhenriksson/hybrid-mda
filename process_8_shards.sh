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

echo "Processing shards $START_SHARD to $END_SHARD for language $LANG"

# Create a temporary job script
JOB_SCRIPT="srun_${LANG}_${START_SHARD}_${END_SHARD}.sh"
LOGS_DIR="slurm-logs"
mkdir -p $LOGS_DIR

cat > $JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=trankit_${LANG}
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=${LOGS_DIR}/%j.out
#SBATCH --error=${LOGS_DIR}/%j.err
#SBATCH --account=project_462000353
#SBATCH --partition=small-g

# Load environment
source venv/bin/activate

# Load PyTorch module
module use /appl/local/csc/modulefiles
module load pytorch/2.0

# Function to process a single shard
process_shard() {
    local lang=\$1
    local shard=\$2
    local gpu_id=\$3
    
    echo "[GPU \${gpu_id}] Processing \${lang} shard \${shard}..."
    
    # Set CUDA device for this task
    export CUDA_VISIBLE_DEVICES=\${gpu_id}
    
    # Run the Python script
    python parse_shard.py \${lang} \${shard} --gpu \${gpu_id}
    
    echo "[GPU \${gpu_id}] Completed \${lang} shard \${shard}"
}

# Process 8 shards in parallel using srun
for i in {0..7}; do
    shard_num=\$((${START_SHARD} + i))
    
    # Check if we've exceeded the end shard
    if [ \$shard_num -gt ${END_SHARD} ]; then
        break
    fi
    
    # Launch task in background
    srun --ntasks=1 --gres=gpu:mi250:1 --cpus-per-task=4 --exact \
         bash -c "process_shard ${LANG} \${shard_num} \$i" &
    
    # Small delay to avoid race conditions
    sleep 1
done

# Wait for all background tasks to complete
wait

echo "All shards ${START_SHARD}-${END_SHARD} for ${LANG} completed!"
EOL

# Make the script executable
chmod +x $JOB_SCRIPT

# Submit the job
echo "Submitting job to process shards $START_SHARD-$END_SHARD for $LANG"
sbatch $JOB_SCRIPT

echo "Job submitted. Check logs directory for progress."