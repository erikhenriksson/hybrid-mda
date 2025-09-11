import ast
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FILTERED_BY_MEDIAN_AND_STD_PATH,
    STATS_AFTER_FILTERING_BY_MEDIAN_AND_STD_PATH,
    TOP_REGISTERS_PATH,
)


def load_and_analyze_stats(stats_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Load stats file and return top 30 register categories plus empty tuple category.

    Returns:
        - List of selected register keys (top 30 + empty tuple)
        - Dictionary mapping register keys to their counts
    """
    try:
        stats_df = pd.read_csv(stats_path, sep="\t")
        print(f"Loaded stats file with {len(stats_df)} categories")
    except Exception as e:
        print(f"Error loading stats file {stats_path}: {str(e)}")
        raise

    # Create register counts dictionary
    register_counts = {}
    empty_tuple_key = None

    for _, row in stats_df.iterrows():
        preds_value = row["preds"]
        count = row["count"]

        # Convert preds to consistent format
        if isinstance(preds_value, str):
            try:
                preds_value = ast.literal_eval(preds_value)
            except (ValueError, SyntaxError):
                # If evaluation fails, keep as string
                pass

        # Convert to tuple for consistency
        if isinstance(preds_value, list):
            preds_value = tuple(preds_value)

        # Convert to string key for consistent lookup
        key = str(preds_value)
        register_counts[key] = count

        # Check for empty tuple
        if preds_value == () or key == "()":
            empty_tuple_key = key

    # Sort by count (descending) and get top 30
    sorted_registers = sorted(register_counts.items(), key=lambda x: x[1], reverse=True)
    top_30_keys = [key for key, count in sorted_registers[:30]]

    print(f"Top 30 registers by frequency:")
    for i, (key, count) in enumerate(sorted_registers[:30], 1):
        print(f"  {i:2d}. {key}: {count:,} examples")

    # Add empty tuple category if it exists
    selected_keys = top_30_keys.copy()
    if empty_tuple_key and empty_tuple_key not in selected_keys:
        selected_keys.append(empty_tuple_key)
        print(
            f"\nAdding empty tuple category: {empty_tuple_key} ({register_counts[empty_tuple_key]:,} examples)"
        )
    elif empty_tuple_key:
        print(f"\nEmpty tuple category already in top 30: {empty_tuple_key}")
    else:
        print(f"\nWarning: No empty tuple category found in stats!")

    return selected_keys, register_counts


def convert_register_key_to_filename(register_key: str) -> str:
    """Convert register key to safe filename."""
    if register_key == "()" or register_key.strip() == "()":
        return "no_register"

    # Remove parentheses, quotes, and convert to filename-safe format
    cleaned = register_key.strip("()").replace("'", "").replace('"', "")
    # Replace commas and spaces with underscores
    cleaned = re.sub(r"[,\s]+", "_", cleaned)
    # Remove any remaining special characters except underscores
    cleaned = re.sub(r"[^\w_]", "", cleaned)

    return cleaned if cleaned else "unknown_register"


def sample_register_data(
    language_code: str,
    selected_keys: List[str],
    register_counts: Dict[str, int],
    sample_size: int = 1000,
) -> Dict[str, int]:
    """
    Sample random examples for each selected register from the language data.

    Returns:
        Dictionary mapping register keys to actual number of samples obtained
    """
    input_path = (
        f"{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_clean_filtered.tsv"
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nProcessing {language_code} data from: {input_path}")

    # First pass: collect all rows by register
    register_data = {key: [] for key in selected_keys}
    chunk_size = 10000
    total_rows = 0

    print("First pass: collecting data by register...")

    try:
        for chunk in tqdm(pd.read_csv(input_path, sep="\t", chunksize=chunk_size)):
            total_rows += len(chunk)

            # Remove unnamed columns
            chunk = chunk[
                [col for col in chunk.columns if not col.startswith("Unnamed: ")]
            ]

            for idx, row in chunk.iterrows():
                # Parse register from fixed_register column and convert to tuple for consistency
                preds_value = row["fixed_register"]
                if isinstance(preds_value, str):
                    preds_value = ast.literal_eval(preds_value)

                # Convert list to tuple for consistent comparison with stats
                if isinstance(preds_value, list):
                    preds_value = tuple(preds_value)

                key = str(preds_value)

                # If this register is in our selected keys, store the row
                if key in register_data:
                    register_data[key].append(row)

    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        raise

    print(f"Processed {total_rows:,} total rows")

    # Second pass: sample and save data for each register
    sampling_results = {}

    # Ensure output directory exists
    output_dir = f"{TOP_REGISTERS_PATH}/{language_code}"
    os.makedirs(output_dir, exist_ok=True)

    print("\nSecond pass: sampling and saving data...")

    for register_key in selected_keys:
        available_rows = len(register_data[register_key])
        filename = convert_register_key_to_filename(register_key)
        output_path = f"{output_dir}/{filename}.tsv"

        if available_rows == 0:
            print(f"WARNING: No data found for register {register_key}")
            sampling_results[register_key] = 0
            continue

        if available_rows < sample_size:
            print(
                f"WARNING: Only {available_rows} examples available for register {register_key} (requested {sample_size})"
            )

        # Sample the data
        sample_count = min(available_rows, sample_size)
        if sample_count == available_rows:
            # Use all available data
            sampled_rows = register_data[register_key]
        else:
            # Random sample
            np.random.seed(42)  # For reproducibility
            sampled_indices = np.random.choice(
                available_rows, size=sample_count, replace=False
            )
            sampled_rows = [register_data[register_key][i] for i in sampled_indices]

        # Convert to DataFrame and save
        if sampled_rows:
            sampled_df = pd.DataFrame(sampled_rows)
            sampled_df.to_csv(output_path, sep="\t", index=False)
            print(
                f"Saved {len(sampled_df)} examples for {register_key} -> {filename}.tsv"
            )

        sampling_results[register_key] = sample_count

    return sampling_results


def process_language_sampling(language_code: str) -> Dict[str, Any]:
    """Process sampling for a specific language."""
    stats_path = f"{STATS_AFTER_FILTERING_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_clean_filtered.tsv"

    try:
        # Load and analyze stats to get top registers
        selected_keys, register_counts = load_and_analyze_stats(stats_path)

        print(f"\nSelected {len(selected_keys)} register categories for sampling")

        # Sample data for each register
        sampling_results = sample_register_data(
            language_code, selected_keys, register_counts
        )

        return {
            "selected_registers": len(selected_keys),
            "sampling_results": sampling_results,
            "total_sampled": sum(sampling_results.values()),
        }

    except Exception as e:
        print(f"Error processing {language_code}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def main():
    """Main function to process all languages."""
    languages = ["fr", "sv"]
    results = {}

    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Processing {lang.upper()} data...")
        print(f"{'=' * 60}")

        try:
            results[lang] = process_language_sampling(lang)
            if "error" not in results[lang]:
                print(f"\n✓ {lang} processed successfully")
                print(f"  - Selected registers: {results[lang]['selected_registers']}")
                print(f"  - Total examples sampled: {results[lang]['total_sampled']:,}")
            else:
                print(f"\n✗ Error processing {lang}: {results[lang]['error']}")
        except Exception as e:
            print(f"\n✗ Unexpected error processing {lang}: {str(e)}")
            results[lang] = {"error": str(e)}

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    for lang, stats in results.items():
        if isinstance(stats, dict) and "error" not in stats:
            print(f"\n{lang.upper()}:")
            print(f"  Total examples sampled: {stats['total_sampled']:,}")

            # Show sampling results for each register
            sampling_results = stats["sampling_results"]
            insufficient_registers = [
                k for k, v in sampling_results.items() if v < 1000
            ]

            if insufficient_registers:
                print(f"  Registers with <1000 examples:")
                for reg_key in insufficient_registers:
                    count = sampling_results[reg_key]
                    filename = convert_register_key_to_filename(reg_key)
                    print(f"    {reg_key} -> {filename}.tsv: {count} examples")
            else:
                print(f"  All registers have 1000 examples ✓")

        else:
            error_msg = (
                stats.get("error", str(stats))
                if isinstance(stats, dict)
                else str(stats)
            )
            print(f"\n{lang.upper()}: Processing failed - {error_msg}")


if __name__ == "__main__":
    main()
