import glob
import os

import pandas as pd
import trankit
from tqdm import tqdm

from config import (
    PARSED_CONLLU_PATH,
    TOP_REGISTERS_PATH,
)


def parse_files_for_language(language_code: str):
    """Parse all register files for a given language using Trankit."""

    # Setup paths
    input_dir = f"{TOP_REGISTERS_PATH}/{language_code}"
    output_dir = f"{PARSED_CONLLU_PATH}/{language_code}"

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Trankit pipeline
    print(f"Initializing Trankit pipeline for {language_code}...")
    p = trankit.Pipeline(language_code, gpu=True)

    # Get all TSV files in the input directory
    tsv_files = glob.glob(f"{input_dir}/*.tsv")

    if not tsv_files:
        print(f"No TSV files found in {input_dir}")
        return

    print(f"Found {len(tsv_files)} files to process")

    # Process each file
    for tsv_file in tqdm(tsv_files, desc=f"Processing {language_code}"):
        filename = os.path.basename(tsv_file)
        register_name = filename.replace(".tsv", "")
        output_file = f"{output_dir}/{register_name}_parsed.tsv"

        print(f"\nProcessing {filename}...")

        try:
            # Load the data
            df = pd.read_csv(tsv_file, sep="\t")

            # Prepare output data
            parsed_rows = []

            # Process each text
            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc=f"Parsing texts in {filename}"
            ):
                text_id = row["id"]
                text_content = str(row["text"])

                # Parse with Trankit
                parsed = p(text_content)

                # Extract tokens from all sentences
                for sent_idx, sentence in enumerate(parsed["sentences"]):
                    for token in sentence["tokens"]:
                        # Define expected columns in order
                        expected_columns = [
                            "id",
                            "text",
                            "lemma",
                            "upos",
                            "xpos",
                            "feats",
                            "head",
                            "deprel",
                            "deps",
                            "misc",
                        ]

                        # Start with text_id and sentence_id
                        parsed_row = {"text_id": text_id, "sentence_id": sent_idx}

                        # Add all expected token fields with defaults for missing keys
                        for col in expected_columns:
                            parsed_row[col] = token.get(col, "")

                        parsed_rows.append(parsed_row)

            # Save parsed data
            if parsed_rows:
                parsed_df = pd.DataFrame(parsed_rows)
                parsed_df.to_csv(output_file, sep="\t", index=False)
                print(f"Saved {len(parsed_rows)} tokens to {output_file}")
            else:
                print(f"No tokens parsed for {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback

            traceback.print_exc()


def main():
    """Main function to process all languages."""
    languages = ["fr", "sv"]

    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Processing {lang.upper()} files...")
        print(f"{'=' * 60}")

        try:
            parse_files_for_language(lang)
            print(f"\n✓ {lang} processing completed")
        except Exception as e:
            print(f"\n✗ Error processing {lang}: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
