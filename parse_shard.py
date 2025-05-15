import argparse
import os
import sys

import pandas as pd
import trankit
from trankit import trankit2conllu

# Language code mapping for Trankit
LANGUAGE_MAP = {"en": "english", "fi": "finnish", "fr": "french", "sv": "swedish"}


def process_shard(language_code, shard_num, gpu_id=None):
    """
    Process a single shard of data using Trankit.

    Args:
        language_code: Two-letter language code (en, fi, fr, sv)
        shard_num: Shard number to process (01-32)
        gpu_id: GPU ID to use (or None for CPU)
    """
    # Set GPU if specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Using GPU {gpu_id} for processing")

    # Map language code to Trankit language name
    trankit_language = LANGUAGE_MAP.get(language_code)

    # Input path (filtered data shard)
    input_path = (
        f"data/shards/{language_code}/{language_code}_shard_{shard_num:02d}.tsv"
    )

    # Output path for parsed data
    output_dir = f"data/parsed/{language_code}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/shard_{shard_num:02d}.conllu"

    print(
        f"Processing {language_code} (shard {shard_num}) with {trankit_language} model"
    )

    # Initialize Trankit pipeline
    p = trankit.Pipeline(trankit_language)

    # Read the entire shard into memory
    df = pd.read_csv(input_path, sep="\t")
    total_rows = len(df)

    # Open output file
    with open(output_path, "w", encoding="utf-8") as f_out:
        # Process each row
        for idx, row in enumerate(df.itertuples()):
            # Print progress
            if idx % 100 == 0:
                progress = (idx / total_rows) * 100
                sys.stdout.write(f"\rProcessing: {idx}/{total_rows} ({progress:.1f}%)")
                sys.stdout.flush()

            # Get text
            text = getattr(row, "text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            # Get register if available (default to 'IN')
            register = (
                getattr(row, "register", "IN") if hasattr(row, "register") else "IN"
            )
            register_lang = f"{register} {language_code}"

            try:
                # Process text with Trankit
                processed = p(text)
                conllu_text = trankit2conllu(processed)

                # Write CONLLU output with headers
                f_out.write(f"###C: NEWDOC\n")
                f_out.write(f"###C: TEXT_ID{idx}\n")
                f_out.write(f"###C: REGISTER={register_lang}\n")
                f_out.write(conllu_text)
                f_out.write("\n")  # Extra new line between documents

            except Exception as e:
                print(f"\nError processing text {idx}: {str(e)[:100]}...")
                continue

    print(f"\nCompleted processing shard {shard_num} for {language_code}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse a shard of filtered text data using Trankit"
    )
    parser.add_argument(
        "language",
        type=str,
        choices=["en", "fi", "fr", "sv"],
        help="Language code to process (en, fi, fr, sv)",
    )
    parser.add_argument(
        "shard", type=int, choices=range(1, 33), help="Shard number to process (1-32)"
    )
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU ID to use (default: None for CPU)"
    )

    args = parser.parse_args()

    try:
        process_shard(args.language, args.shard, args.gpu)
        print(f"✓ Successfully processed {args.language} shard {args.shard}")
    except Exception as e:
        print(f"✗ Error processing {args.language} shard {args.shard}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
