import ast
import os
import sys

import numpy as np
import pandas as pd

from config import (
    FILTERED_BY_MEDIAN_AND_STD_TEMPLATE,
    STATS_PATH_TEMPLATE,
)


def standardize_pred_format(pred_value):
    """
    Standardize prediction format to ensure consistent comparison
    between lists ['AV'] and tuples ('AV',)

    Args:
        pred_value: The prediction value as a string

    Returns:
        Standardized string representation
    """
    try:
        # Try to parse the string
        if isinstance(pred_value, str):
            # Handle both list and tuple formats
            if pred_value.startswith("[") and pred_value.endswith("]"):
                # Convert to tuple for consistent comparison
                parsed = tuple(ast.literal_eval(pred_value))
                return str(parsed)
            elif pred_value.startswith("(") and pred_value.endswith(")"):
                # Already a tuple format
                return pred_value
            else:
                # Try to parse with ast.literal_eval
                parsed = ast.literal_eval(pred_value)
                if isinstance(parsed, list):
                    return str(tuple(parsed))
                elif isinstance(parsed, tuple):
                    return str(parsed)
                else:
                    return str(parsed)
        else:
            # If it's already a list or tuple
            if isinstance(pred_value, list):
                return str(tuple(pred_value))
            elif isinstance(pred_value, tuple):
                return str(pred_value)
            else:
                return str(pred_value)
    except:
        # Return as is if parsing fails
        return str(pred_value)


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

        # Filtered stats path - UPDATED from "analysis" to "reports"
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

            # Standardize the preds format in both datasets
            original_stats["preds_standardized"] = original_stats["preds"].apply(
                standardize_pred_format
            )
            filtered_stats["preds_standardized"] = filtered_stats["preds"].apply(
                standardize_pred_format
            )

            # Create merged dataset using standardized preds
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

    # Keep original preds column
    original_preds = original_stats["preds"].copy()

    # Rename all columns except preds_standardized
    original_stats_renamed = original_stats.drop(columns=["preds"]).copy()
    original_stats_renamed.columns = [
        col if col == "preds_standardized" else f"{col}_original"
        for col in original_stats_renamed.columns
    ]

    filtered_stats_renamed = filtered_stats.drop(columns=["preds"]).copy()
    filtered_stats_renamed.columns = [
        col if col == "preds_standardized" else f"{col}_filtered"
        for col in filtered_stats_renamed.columns
    ]

    # Merge on standardized prediction category
    merged = pd.merge(
        original_stats_renamed,
        filtered_stats_renamed,
        on="preds_standardized",
        how="outer",
    )

    # Rename standardized column back to preds and use original format
    merged["preds"] = merged["preds_standardized"]
    merged = merged.drop(columns=["preds_standardized"])

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

    # Reorder columns to put preds first
    cols = merged.columns.tolist()
    cols.remove("preds")
    cols = ["preds"] + cols
    merged = merged[cols]

    return merged


def main():
    # Process all language files
    compare_stats()


if __name__ == "__main__":
    main()
