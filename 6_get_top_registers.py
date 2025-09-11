import ast
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from config import (
    FILTERED_BY_MEDIAN_AND_STD_PATH,
    STATS_AFTER_FILTERING_BY_MEDIAN_AND_STD_PATH,
    TOP_REGISTERS_PATH,
)


def load_and_get_top_registers(stats_path: str) -> List[str]:
    """
    Load stats file and return top 30 register categories plus empty tuple category.

    Returns:
        List of selected register keys (top 30 + empty tuple)
    """
    try:
        stats_df = pd.read_csv(stats_path, sep="\t")
        print(f"Loaded stats file with {len(stats_df)} categories")
    except Exception as e:
        print(f"Error loading stats file {stats_path}: {str(e)}")
        raise

    # Create register counts list for sorting
    register_data = []
    empty_tuple_key = None

    for _, row in stats_df.iterrows():
        preds_value = row["preds"]
        count = row["count"]

        # Parse tuple from string format
        if isinstance(preds_value, str):
            preds_value = ast.literal_eval(preds_value)

        # Use the tuple directly as key (convert to string for dict lookup)
        key = str(preds_value)
        register_data.append((key, count))

        # Check for empty tuple
        if preds_value == () or key == "()":
            empty_tuple_key = key

    # Sort by count (descending) and get top 30
    register_data.sort(key=lambda x: x[1], reverse=True)
    top_30_keys = [key for key, count in register_data[:30]]

    print(f"Top 30 registers by frequency:")
    for i, (key, count) in enumerate(register_data[:30], 1):
        print(f"  {i:2d}. {key}: {count:,} examples")

    # Add empty tuple category if it exists and not already in top 30
    selected_keys = top_30_keys.copy()
    if empty_tuple_key and empty_tuple_key not in selected_keys:
        selected_keys.append(empty_tuple_key)
        empty_count = next(
            (count for key, count in register_data if key == empty_tuple_key), 0
        )
        print(
            f"\nAdding empty tuple category: {empty_tuple_key} ({empty_count:,} examples)"
        )
    elif empty_tuple_key:
        print(f"\nEmpty tuple category already in top 30: {empty_tuple_key}")
    else:
        print(f"\nWarning: No empty tuple category found in stats!")

    return selected_keys


def convert_register_key_to_filename(register_key: str) -> str:
    """Convert register key to safe filename."""
    if register_key == "()" or register_key.strip() == "()":
        return "no_register"

    # Remove parentheses, quotes, and convert to filename-safe format
    cleaned = register_key.strip("()").replace("'", "").replace('"', "")
    # Replace commas and spaces with underscores
    import re

    cleaned = re.sub(r"[,\s]+", "_", cleaned)
    # Remove any remaining special characters except underscores
    cleaned = re.sub(r"[^\w_]", "", cleaned)
    # Remove trailing underscores
    cleaned = cleaned.rstrip("_")

    return cleaned if cleaned else "unknown_register"


def sample_registers_incrementally(
    language_code: str, selected_keys: List[str], sample_size: int = 1000
) -> Dict[str, int]:
    """
    Sample examples incrementally - take first N examples as we encounter them.

    Returns:
        Dictionary mapping register keys to actual number of samples obtained
    """
    input_path = (
        f"{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_clean_filtered.tsv"
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\nProcessing {language_code} data from: {input_path}")

    # Setup output directory and files
    output_dir = f"{TOP_REGISTERS_PATH}/{language_code}"
    os.makedirs(output_dir, exist_ok=True)

    # Track counts and open file handles
    register_counts = {key: 0 for key in selected_keys}
    output_files = {}
    headers_written = {}

    # Open all output files
    for register_key in selected_keys:
        filename = convert_register_key_to_filename(register_key)
        output_path = f"{output_dir}/{filename}.tsv"
        output_files[register_key] = open(output_path, "w")
        headers_written[register_key] = False

    chunk_size = 10000
    total_rows = 0
    completed_registers = set()

    try:
        print("Processing data incrementally...")

        for chunk in tqdm(pd.read_csv(input_path, sep="\t", chunksize=chunk_size)):
            total_rows += len(chunk)

            # Remove unnamed columns
            chunk = chunk[
                [col for col in chunk.columns if not col.startswith("Unnamed: ")]
            ]

            # Get header from first chunk
            if not any(headers_written.values()):
                header_line = "\t".join(chunk.columns) + "\n"

            for _, row in chunk.iterrows():
                # Parse register from fixed_register column and convert to tuple for consistency
                preds_value = row["fixed_register"]
                if isinstance(preds_value, str):
                    preds_value = ast.literal_eval(preds_value)

                # Convert list to tuple for consistent comparison with stats
                if isinstance(preds_value, list):
                    preds_value = tuple(preds_value)

                key = str(preds_value)

                # If this register is in our selected keys and not yet complete
                if key in register_counts and key not in completed_registers:
                    # Write header if not written yet
                    if not headers_written[key]:
                        output_files[key].write(header_line)
                        headers_written[key] = True

                    # Write the row
                    row_line = "\t".join(str(val) for val in row.values) + "\n"
                    output_files[key].write(row_line)

                    register_counts[key] += 1

                    # Mark as complete if we've reached the sample size
                    if register_counts[key] >= sample_size:
                        completed_registers.add(key)
                        output_files[key].close()
                        filename = convert_register_key_to_filename(key)
                        print(
                            f"Completed {key} -> {filename}.tsv: {register_counts[key]} examples"
                        )

            # Check if all registers are complete
            if len(completed_registers) == len(selected_keys):
                print(f"All registers completed! Processed {total_rows:,} rows total.")
                break

        else:
            # Loop completed without breaking - some registers may be incomplete
            print(f"Finished processing all {total_rows:,} rows.")

            # Close any remaining open files
            for key, file_handle in output_files.items():
                if key not in completed_registers:
                    file_handle.close()

            # Report incomplete registers
            incomplete_registers = [
                key for key in selected_keys if register_counts[key] < sample_size
            ]
            if incomplete_registers:
                print(
                    f"\nWARNING: The following registers have fewer than {sample_size} examples:"
                )
                for key in incomplete_registers:
                    filename = convert_register_key_to_filename(key)
                    print(f"  {key} -> {filename}.tsv: {register_counts[key]} examples")

    finally:
        # Ensure all files are closed
        for file_handle in output_files.values():
            if not file_handle.closed:
                file_handle.close()

    return register_counts


def process_language_sampling(language_code: str) -> Dict[str, Any]:
    """Process sampling for a specific language."""
    stats_path = f"{STATS_AFTER_FILTERING_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_clean_filtered.tsv"

    try:
        # Load and analyze stats to get top registers
        selected_keys = load_and_get_top_registers(stats_path)

        print(f"\nSelected {len(selected_keys)} register categories for sampling")

        # Sample data for each register incrementally
        sampling_results = sample_registers_incrementally(language_code, selected_keys)

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
