import os
import sys  # Add missing import
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (  # Import config
    DATA_PATH_TEMPLATE,
    FILTERED_BY_MEDIAN_AND_STD_TEMPLATE,
    STATS_PATH_TEMPLATE,
)


def analyze_text_lengths(input_dir, output_dir):
    """
    Analyze text length distributions (in words) for processed files and save in separate language files.

    Args:
        input_dir: Directory containing the processed TSV files
        output_dir: Directory where to save the analysis results
    """
    # Languages to process
    languages = ["en", "fi", "fr", "sv"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for lang in languages:
        print(f"\nAnalyzing {lang} data...")

        # Path to processed file - update to use the OUTPUT_PATH_TEMPLATE from config
        file_path = FILTERED_BY_MEDIAN_AND_STD_TEMPLATE.format(lang)
        # Define a new output path for the analysis results
        output_path = os.path.join(output_dir, f"{lang}_embeds_analysis.tsv")

        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        # Track statistics by prediction category
        pred_stats = defaultdict(list)

        # Process in chunks to handle large files
        chunk_size = 10000
        total_rows = 0

        try:
            # Process data in chunks
            for chunk in tqdm(pd.read_csv(file_path, sep="\t", chunksize=chunk_size)):
                total_rows += len(chunk)

                # Convert preds column if it's a string using the same method as the first code
                if "preds" in chunk.columns:
                    # Use the convert_preds_to_key function logic for consistency
                    def convert_preds(preds_value):
                        if isinstance(preds_value, str):
                            try:
                                preds_value = eval(preds_value)
                            except:
                                pass
                        # Convert to tuple for hashability if it's a list
                        if isinstance(preds_value, list):
                            preds_value = tuple(preds_value)
                        return preds_value

                    chunk["preds"] = chunk["preds"].apply(convert_preds)

                    # Calculate word counts
                    chunk["word_count"] = chunk["text"].apply(
                        lambda x: len(str(x).split())
                    )

                    # Group by prediction category using the same key format as the first code
                    for idx, row in chunk.iterrows():
                        preds_key = str(row["preds"])
                        pred_stats[preds_key].append(row["word_count"])

                # Print progress
                sys.stdout.write(f"\rProcessed {total_rows} rows")
                sys.stdout.flush()

            print(
                f"\nCompleted analysis for {lang}, found {len(pred_stats)} prediction categories"
            )

            # Calculate statistics for each prediction category
            results = []

            for preds_key, word_counts in pred_stats.items():
                # Calculate statistics
                count = len(word_counts)
                median = np.median(word_counts) if count > 0 else 0
                mean = np.mean(word_counts) if count > 0 else 0
                std = np.std(word_counts) if count > 0 else 0
                min_count = np.min(word_counts) if count > 0 else 0
                max_count = np.max(word_counts) if count > 0 else 0

                # Store results
                results.append(
                    {
                        "preds": preds_key,  # Keep as string for consistency with first code
                        "count": count,
                        "median": median,
                        "mean": mean,
                        "std": std,
                        "min": min_count,
                        "max": max_count,
                    }
                )

            # Create DataFrame from results
            results_df = pd.DataFrame(results)

            # Save to TSV for this language
            results_df.to_csv(output_path, sep="\t", index=False)
            print(f"Analysis for {lang} saved to {output_path}")

            # Generate histograms for the top 5 most common prediction categories
            top_categories = sorted(
                pred_stats.items(), key=lambda x: len(x[1]), reverse=True
            )[:5]

            plt.figure(figsize=(15, 10))
            for i, (preds_key, word_counts) in enumerate(top_categories):
                plt.subplot(2, 3, i + 1)
                plt.hist(word_counts, bins=50, alpha=0.7)

                # Try to make the title more readable
                title = preds_key
                if len(title) > 20:
                    title = title[:18] + "..."

                plt.title(f"{lang}: {title}")
                plt.xlabel("Word Count")
                plt.ylabel("Frequency")

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{lang}_word_distributions.png"))
            plt.close()

        except Exception as e:
            print(f"Error analyzing {lang} data: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nAnalysis complete. Results saved to {output_dir}")


def main():
    # Update paths to use the config file approach
    output_dir = (
        os.path.dirname(FILTERED_BY_MEDIAN_AND_STD_TEMPLATE.format("en")) + "/analysis"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis - no need to pass input_dir since we're using OUTPUT_PATH_TEMPLATE
    analyze_text_lengths(None, output_dir)


if __name__ == "__main__":
    main()
