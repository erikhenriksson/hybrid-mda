import glob
import os

import pandas as pd
import trankit
from tqdm import tqdm

from config import (
    PARSED_CONLLU_PATH,
    TOP_REGISTERS_PATH,
)


def check_parsing_status(original_file: str, parsed_file: str) -> str:
    """
    Check if parsing is complete by comparing text IDs.

    Returns:
        'complete' - parsing is done, skip this file
        'partial' - parsing is incomplete, delete parsed file and restart
        'missing' - no parsed file exists, start fresh
    """
    if not os.path.exists(parsed_file):
        return "missing"

    try:
        # Load original file IDs
        original_df = pd.read_csv(original_file, sep="\t")
        original_ids = set(original_df["id"])

        # Load parsed file IDs
        parsed_df = pd.read_csv(parsed_file, sep="\t")
        parsed_ids = set(parsed_df["text_id"])

        if original_ids == parsed_ids:
            return "complete"
        else:
            return "partial"

    except Exception as e:
        print(f"Error checking parsing status: {e}")
        return "partial"  # Safer to restart if we can't determine status


def parse_files_for_language(language_code: str):
    """Parse all register files for a given language using Trankit."""

    # Map language codes to Trankit language names
    trankit_lang_map = {
        "fr": "french",
        "sv": "swedish",  # Try swedish first, may need adjustment
    }

    trankit_lang = trankit_lang_map.get(language_code, language_code)

    # Setup paths
    input_dir = f"{TOP_REGISTERS_PATH}/{language_code}"
    output_dir = f"{PARSED_CONLLU_PATH}/{language_code}"

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Trankit pipeline
    print(f"Initializing Trankit pipeline for {trankit_lang} (from {language_code})...")
    try:
        p = trankit.Pipeline(trankit_lang, gpu=True)
    except Exception as e:
        print(f"Error initializing Trankit for '{trankit_lang}': {e}")
        print(
            "Available languages can be checked with: import trankit; print(trankit.supported_langs)"
        )
        return

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

        # Check if parsing is already complete
        status = check_parsing_status(tsv_file, output_file)

        if status == "complete":
            print(f"✓ {filename} already parsed completely, skipping")
            continue
        elif status == "partial":
            print(f"⚠ {filename} partially parsed, restarting from scratch")
            try:
                os.remove(output_file)
            except:
                pass
        else:  # status == 'missing'
            print(f"→ {filename} starting fresh parse")

        print(f"Processing {filename}...")

        try:
            # Load the data
            df = pd.read_csv(tsv_file, sep="\t")

            # Extract register info from filename (convert underscores to spaces)
            register_info = register_name.replace("_", " ")

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

                        # Start with text_id, sentence_id, and register
                        parsed_row = {
                            "text_id": text_id,
                            "sentence_id": sent_idx,
                            "register": register_info,
                        }

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
