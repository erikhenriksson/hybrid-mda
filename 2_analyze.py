import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# We'll need to import the template from the same config module
from config import FILTERED_BY_MEDIAN_AND_STD_TEMPLATE


def generate_filtered_stats(filtered_dir, language_codes=None):
    """
    Generate statistics from filtered data files.

    Args:
        filtered_dir: Base directory containing filtered data files
        language_codes: List of language codes to process, defaults to ['en', 'fi', 'fr', 'sv']
    """
    if language_codes is None:
        language_codes = ["en", "fi", "fr", "sv"]

    for lang in language_codes:
        print(f"\nProcessing filtered data for {lang}...")

        # Construct filtered data path
        filtered_path = os.path.join(
            filtered_dir, FILTERED_BY_MEDIAN_AND_STD_TEMPLATE.format(lang)
        )

        if not os.path.exists(filtered_path):
            print(f"Filtered file not found: {filtered_path}")
            continue

        # Output path for statistics
        output_stats_path = os.path.join(
            "reports", FILTERED_BY_MEDIAN_AND_STD_TEMPLATE.format(lang)
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
                preds_value = row["preds"]

                # Handle different formats
                if isinstance(preds_value, str):
                    try:
                        preds_value = eval(preds_value)
                    except:
                        pass

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

            # Calculate standard deviation if we have more than one sample
            std = np.std(lengths) if count > 1 else None

            # Append to statistics data
            stats_data.append(
                {
                    "preds": preds_key,
                    "count": count,
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
    # Base directory containing filtered data files
    filtered_dir = "./data"

    # Process all language files
    generate_filtered_stats(filtered_dir)

    print("\nProcessing Complete")


if __name__ == "__main__":
    main()
