import os

# Set cache directories FIRST - before importing torch/transformers/detoxify
os.environ['TORCH_HOME'] = '/scratch/project_2002026/ehenriks/.cache'
os.environ['HF_HOME'] = '/scratch/project_2002026/ehenriks/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/project_2002026/ehenriks/.cache'
os.environ['HF_DATASETS_CACHE'] = '/scratch/project_2002026/ehenriks/.cache'

# Create cache directory if it doesn't exist
os.makedirs('/scratch/project_2002026/ehenriks/.cache', exist_ok=True)

import re
import sys

import pandas as pd
from tqdm import tqdm

from config import (
    FILTERED_BY_MIN_LENGTH_PATH,
    MIN_TEXT_LENGTH,
    RAW_DATA_PATH,
)


def process_language_data(language_code):
    """Process data for a specific language in chunks, filtering by minimum text length."""
    # Updated paths
    data_path = f"{RAW_DATA_PATH}/{language_code}_embeds.tsv"
    output_path = (
        f"data/{FILTERED_BY_MIN_LENGTH_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process in chunks
    chunk_size = 10000  # Adjust based on available memory
    total_rows = 0
    kept_rows = 0

    # Create output file and write header
    try:
        # First, get the header without the index column
        header_df = pd.read_csv(data_path, sep="\t", nrows=0)
        # Filter out the 'Unnamed: 0' column if it exists
        header = [col for col in header_df.columns if not col.startswith("Unnamed: ")]

        # Open output file
        with open(output_path, "w") as f_out:
            # Write header
            f_out.write("\t".join(header) + "\n")

            # Process data in chunks
            for chunk in tqdm(pd.read_csv(data_path, sep="\t", chunksize=chunk_size)):
                total_rows += len(chunk)

                # Remove unnamed columns
                chunk = chunk[
                    [col for col in chunk.columns if not col.startswith("Unnamed: ")]
                ]

                # Create a copy to avoid SettingWithCopyWarning
                chunk_copy = chunk.copy()

                # Prepare data for filtering
                keep_rows = []

                for idx, row in chunk_copy.iterrows():
                    # Clean text
                    chunk_copy.at[idx, "text"] = re.sub(
                        r"\s+", " ", str(row["text"])
                    ).strip()

                    # Calculate text length
                    text_length = len(str(row["text"]).split())

                    # Determine if row should be kept based on minimum text length
                    keep_rows.append(text_length >= MIN_TEXT_LENGTH)

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
