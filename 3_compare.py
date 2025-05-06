import os
import sys

import numpy as np
import pandas as pd

from config import (
    FILTERED_BY_MEDIAN_AND_STD_TEMPLATE,
    STATS_PATH_TEMPLATE,
)


def compare_stats(language_codes=None):
    """
    Compare original and filtered statistics.

    Args:
        language_codes: List of language codes to process, defaults to ['en', 'fi', 'fr', 'sv']
    """
    if language_codes is None:
        language_codes = ["en", "fi", "fr", "sv"]

    # Create template for output files
    COMPARISON_TEMPLATE = "1_filtered_by_median_and_std/{}_embeds_comparison.tsv"

    # Base output directory for comparison results
    output_dir = "reports"

    for lang in language_codes:
        # Original stats path
        original_stats_path = STATS_PATH_TEMPLATE.format(lang)

        # Filtered stats path
        filtered_stats_path = os.path.join(
            "reports", FILTERED_BY_MEDIAN_AND_STD_TEMPLATE.format(lang)
        )

        # Output path for comparison
        comparison_path = os.path.join(output_dir, COMPARISON_TEMPLATE.format(lang))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)

        # Check if both files exist
        if not os.path.exists(original_stats_path) or not os.path.exists(
            filtered_stats_path
        ):
            continue

        # Load stats files
        try:
            original_stats = pd.read_csv(original_stats_path, sep="\t")
            filtered_stats = pd.read_csv(filtered_stats_path, sep="\t")

            # Create merged dataset
            merged_df = create_merged_dataset(original_stats, filtered_stats)

            # Sort by preds alphabetically
            merged_df = merged_df.sort_values("preds")

            # Save to the specified path
            merged_df.to_csv(comparison_path, sep="\t", index=False)

        except Exception as e:
            import traceback

            traceback.print_exc()


def create_merged_dataset(original_stats, filtered_stats):
    """
    Create a merged dataset with both original and filtered stats.

    Args:
        original_stats: DataFrame with original statistics
        filtered_stats: DataFrame with filtered statistics

    Returns:
        DataFrame with merged statistics
    """
    # Rename columns to indicate source
    original_stats = original_stats.copy()
    filtered_stats = filtered_stats.copy()

    original_stats.columns = [
        col if col == "preds" else f"{col}_original" for col in original_stats.columns
    ]

    filtered_stats.columns = [
        col if col == "preds" else f"{col}_filtered" for col in filtered_stats.columns
    ]

    # Merge on prediction category
    merged = pd.merge(original_stats, filtered_stats, on="preds", how="outer")

    # Add subtraction columns
    # For counts
    if "count_original" in merged.columns and "count_filtered" in merged.columns:
        merged["count_diff"] = merged["count_original"] - merged[
            "count_filtered"
        ].fillna(0)

    # For medians
    if "median_original" in merged.columns and "median_filtered" in merged.columns:
        merged["median_diff"] = merged["median_original"] - merged[
            "median_filtered"
        ].fillna(0)

    # For standard deviations
    if "std_original" in merged.columns and "std_filtered" in merged.columns:
        # Convert empty strings to NaN
        merged["std_original"] = pd.to_numeric(merged["std_original"], errors="coerce")
        merged["std_filtered"] = pd.to_numeric(merged["std_filtered"], errors="coerce")

        merged["std_diff"] = merged["std_original"] - merged["std_filtered"].fillna(0)

    return merged


def main():
    # Process all language files
    compare_stats()


if __name__ == "__main__":
    main()
