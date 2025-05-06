import os
import re
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FILTERED_BY_MEDIAN_AND_STD_PATH,
    FILTERED_BY_MIN_LENGTH_PATH,
)


def load_stats_file(stats_path):
    """Load the statistics file and prepare the stats dictionary."""
    stats_df = pd.read_csv(stats_path, sep="\t")

    # Create a dictionary mapping prediction categories to their stats
    stats_dict = {}
    for _, row in stats_df.iterrows():
        # Convert preds to a consistent string format
        preds_value = row["preds"]
        if isinstance(preds_value, str):
            try:
                preds_value = eval(preds_value)
            except:
                print("error")
                pass

        # Handle list or tuple type
        if isinstance(preds_value, (list, tuple)):
            preds_key = tuple(preds_value)  # Convert to tuple for hashability
        else:
            preds_key = preds_value

        # Convert to string for consistent lookup
        preds_key_str = str(preds_key)

        stats_dict[preds_key_str] = {
            "mean": row["mean"],
            "std": row.get("std", 0),  # Handle cases where std might be missing
        }

    return stats_dict


def convert_preds_to_key(preds_value):
    """Convert preds value to a consistent string key format."""
    # Convert from string to Python object if needed
    if isinstance(preds_value, str):
        try:
            preds_value = eval(preds_value)
        except:
            pass

    # Convert to tuple for hashability if it's a list
    if isinstance(preds_value, list):
        preds_value = tuple(preds_value)

    # Convert to string for consistent lookup
    return str(preds_value)


def process_language_data(language_code):
    """Process data for a specific language in chunks with mean +- 1 std filtering."""
    # Input path (already filtered by minimum length)
    input_path = (
        f"data/{FILTERED_BY_MIN_LENGTH_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Output path for mean ± std filtering
    output_path = (
        f"data/{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Stats path (generated from min length filtered data)
    stats_path = (
        f"reports/{FILTERED_BY_MIN_LENGTH_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load stats
    try:
        stats_dict = load_stats_file(stats_path)
        print(
            f"Loaded statistics for {language_code}, found {len(stats_dict)} prediction categories"
        )
    except Exception as e:
        print(f"Error loading stats for {language_code}: {str(e)}")
        return {"error": str(e)}

    # Process in chunks
    chunk_size = 10000  # Adjust based on available memory
    total_rows = 0
    kept_rows = 0
    missing_categories = set()

    # Create output file and write header
    try:
        # First, get the header without the index column
        header_df = pd.read_csv(input_path, sep="\t", nrows=0)
        # Filter out the 'Unnamed: 0' column if it exists
        header = [col for col in header_df.columns if not col.startswith("Unnamed: ")]

        # Open output file
        with open(output_path, "w") as f_out:
            # Write header
            f_out.write("\t".join(header) + "\n")

            # Process data in chunks
            for chunk in tqdm(pd.read_csv(input_path, sep="\t", chunksize=chunk_size)):
                total_rows += len(chunk)

                # Remove unnamed columns
                chunk = chunk[
                    [col for col in chunk.columns if not col.startswith("Unnamed: ")]
                ]

                # Create a copy to avoid SettingWithCopyWarning
                chunk_copy = chunk.copy()

                # Check for missing categories in this chunk and prepare data for filtering
                keep_rows = []

                for idx, row in chunk_copy.iterrows():
                    # Convert preds to a consistent key format
                    preds_value = row["preds"]
                    preds_key = convert_preds_to_key(preds_value)

                    # Check if category is missing
                    if (
                        preds_key not in stats_dict
                        and preds_key not in missing_categories
                    ):
                        missing_categories.add(preds_key)

                    # Clean text
                    chunk_copy.at[idx, "text"] = re.sub(
                        r"\s+", " ", str(row["text"])
                    ).strip()

                    # Calculate text length
                    text_length = len(str(row["text"]).split())

                    # Determine if row should be kept
                    if preds_key in stats_dict:
                        mean = stats_dict[preds_key]["mean"]
                        std = stats_dict[preds_key]["std"]
                        lower_bound = max(0, mean - std)
                        upper_bound = mean + std
                        keep_rows.append(lower_bound <= text_length <= upper_bound)
                    else:
                        keep_rows.append(False)

                # If any missing categories are found, abort
                if missing_categories:
                    error_msg = f"ERROR: The following prediction categories are in the data file but missing from the stats file: {', '.join(missing_categories)}"
                    print(error_msg)
                    print(
                        "Exiting script to avoid silent continuation with incomplete data."
                    )
                    sys.exit(1)  # Exit with error code

                # Filter chunk based on text length criteria
                chunk_copy["keep"] = keep_rows
                filtered_chunk = chunk_copy[chunk_copy["keep"]]

                kept_rows += len(filtered_chunk)

                # Remove temporary columns and write to file
                filtered_chunk = filtered_chunk.drop(columns=["keep"])
                filtered_chunk.to_csv(f_out, sep="\t", header=False, index=False)

                # Print progress
                sys.stdout.write(
                    f"\rProcessed {total_rows} rows, kept {kept_rows} rows"
                )
                sys.stdout.flush()

    except Exception as e:
        print(f"Error processing data for {language_code}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    # Return stats
    return {
        "original_rows": total_rows,
        "filtered_rows": kept_rows,
        "percentage_kept": round(kept_rows / total_rows * 100, 2)
        if total_rows > 0
        else 0,
    }


def main():
    # Process all language files
    languages = ["en", "fi", "fr", "sv"]
    results = {}

    for lang in languages:
        print(f"\nProcessing {lang} data...")
        try:
            results[lang] = process_language_data(lang)
            print(f"\n✓ {lang} processed successfully")
        except SystemExit:
            # This happens when we deliberately exit due to missing categories
            print(
                f"\n✗ Processing of {lang} halted due to missing prediction categories."
            )
            break
        except Exception as e:
            print(f"\n✗ Error processing {lang}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\nProcessing Summary:")
    print("=================")
    for lang, stats in results.items():
        if isinstance(stats, dict) and "error" not in stats:
            print(
                f"{lang}: {stats['filtered_rows']}/{stats['original_rows']} rows kept ({stats['percentage_kept']}%)"
            )
        else:
            error_msg = (
                stats.get("error", str(stats))
                if isinstance(stats, dict)
                else str(stats)
            )
            print(f"{lang}: Processing failed - {error_msg}")


if __name__ == "__main__":
    main()
