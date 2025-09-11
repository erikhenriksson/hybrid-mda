import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_filtered_stats(input_dir, language_codes=None):
    """
    Generate statistics from filtered data files.

    Args:
        input_dir: Directory containing filtered data files
        language_codes: List of language codes to process, defaults to ['fr', 'sv']
    """

    # Extract the base directory name to use in output path
    base_dir_name = os.path.basename(input_dir)

    for lang in language_codes:
        print(f"\nProcessing filtered data for {lang}...")

        # Construct filtered data path
        filtered_path = os.path.join(input_dir, f"{lang}_embeds_clean.tsv")

        if not os.path.exists(filtered_path):
            print(f"Filtered file not found: {filtered_path}")
            continue

        # Output path for statistics - keep same directory structure
        output_stats_path = os.path.join(
            "reports", base_dir_name, f"{lang}_embeds_clean.tsv"
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

        # Process the data and generate statistics
        process_filtered_file(filtered_path, output_stats_path, lang)

        print(f"✓ Filtered statistics for {lang} saved to {output_stats_path}")


def process_filtered_file(filtered_path, output_stats_path, lang):
    """
    Process a filtered file and generate statistics.

    Args:
        filtered_path: Path to the filtered data file
        output_stats_path: Path to save the statistics
        lang: Language code
    """
    # Dictionary to store statistics for each prediction category
    preds_stats = defaultdict(list)

    # Process in chunks to manage memory
    chunk_size = 10000
    total_rows = 0

    try:
        # Process data in chunks
        for chunk in tqdm(pd.read_csv(filtered_path, sep="\t", chunksize=chunk_size)):
            total_rows += len(chunk)

            # Process each row in the chunk
            for _, row in chunk.iterrows():
                # Get prediction category and convert to a consistent format
                preds_value = row["fixed_register"]

                # Handle different formats
                if isinstance(preds_value, str):
                    try:
                        preds_value = eval(preds_value)
                    except:
                        print(f"Warning: Could not evaluate preds_value: {preds_value}")
                        exit()

                # Convert to tuple for hashability if it's a list
                if isinstance(preds_value, list):
                    preds_value = tuple(preds_value)

                # Convert to string for consistent lookup
                preds_key = str(preds_value)

                # Calculate text length (word count)
                text_length = len(str(row["text"]).split())

                # Add length to stats for this prediction category
                preds_stats[preds_key].append(text_length)

            # Print progress
            sys.stdout.write(f"\rProcessed {total_rows} rows")
            sys.stdout.flush()

        print(
            f"\nCalculating statistics for {len(preds_stats)} prediction categories..."
        )

        # Create DataFrame for statistics
        stats_data = []
        for preds_key, lengths in preds_stats.items():
            # Calculate statistics
            count = len(lengths)
            median = np.median(lengths)
            mean = np.mean(lengths)
            min_length = np.min(lengths)
            max_length = np.max(lengths)

            # Calculate standard deviation if we have more than one sample
            std = np.std(lengths) if count > 1 else None

            # Append to statistics data
            stats_data.append(
                {
                    "preds": preds_key,
                    "count": count,
                    "min": min_length,
                    "max": max_length,
                    "mean": mean,
                    "median": median,
                    "std": std if std is not None else "",
                }
            )

        # Create DataFrame and sort by count (descending)
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values("count", ascending=False)

        # Save to file
        stats_df.to_csv(output_stats_path, sep="\t", index=False)

        print(f"Statistics saved with {len(stats_df)} prediction categories")
        print(f"Total rows processed: {total_rows}")

    except Exception as e:
        print(f"Error processing filtered data for {lang}: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Generate statistics from filtered text data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing filtered data files (e.g., data/filtered_by_min_length)",
    )

    args = parser.parse_args()

    # Default language codes
    language_codes = ["fr", "sv"]

    # Process all language files
    generate_filtered_stats(args.input_dir, language_codes)

    print("\nProcessing Complete")


if __name__ == "__main__":
    main()
