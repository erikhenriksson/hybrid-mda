#!/bin/bash

# Check if language parameter is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <language_code>"
    echo "Supported language codes: en, fi, fr, sv"
    exit 1
fi

LANG=$1

# Validate language code
if [[ ! "$LANG" =~ ^(en|fi|fr|sv)$ ]]; then
    echo "Error: Invalid language code '$LANG'"
    echo "Supported language codes: en, fi, fr, sv"
    exit 1
fi

# Source and destination paths
# Update these paths according to your config
SRC_DIR="data/filtered_by_median_and_std"
SHARD_DIR="data/shards/$LANG"
SRC_FILE="$SRC_DIR/${LANG}_embeds_filtered.tsv"

# Create directories
mkdir -p "$SHARD_DIR"

echo "Sharding $LANG data file: $SRC_FILE"

# Check if source file exists
if [ ! -f "$SRC_FILE" ]; then
    echo "Error: Source file not found: $SRC_FILE"
    exit 1
fi

# First, extract the header
echo "Extracting header..."
head -n 1 "$SRC_FILE" > "header_$LANG.tsv"

# Count total lines to calculate progress
total_lines=$(wc -l < "$SRC_FILE")
data_lines=$((total_lines - 1))
echo "Total lines in file: $total_lines (excluding header: $data_lines)"

# Create a temporary directory for the intermediate files
TMP_DIR="tmp_shards_$LANG"
mkdir -p "$TMP_DIR"

# Split the file (excluding header) into 32 chunks
echo "Splitting file into 32 shards..."
tail -n +2 "$SRC_FILE" | split -n l/32 --numeric-suffixes=1 --suffix-length=2 - "$TMP_DIR/shard_${LANG}_"

# Add the header to each shard
echo "Adding header to each shard..."
for i in $(seq -f "%02g" 1 32); do
    shardfile="$TMP_DIR/shard_${LANG}_$i"
    if [ -f "$shardfile" ]; then
        outfile="$SHARD_DIR/${LANG}_shard_$i.tsv"
        cat "header_$LANG.tsv" "$shardfile" > "$outfile"
        shard_lines=$(wc -l < "$outfile")
        echo "Created shard $i: $outfile ($shard_lines lines)"
    else
        echo "Warning: Shard file not found: $shardfile"
    fi
done

# Clean up
echo "Cleaning up temporary files..."
rm "header_$LANG.tsv"
rm -rf "$TMP_DIR"

echo "Sharding complete. All shards saved to $SHARD_DIR/"