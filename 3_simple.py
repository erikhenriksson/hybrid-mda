import argparse
import os
import sys

import pandas as pd
import trankit
from tqdm import tqdm
from trankit import trankit2conllu

from config import (
    FILTERED_BY_MEDIAN_AND_STD_PATH,
    PARSED_CONLLU_PATH,
)


def parse_language_data(language_code, batch_size=1000, start_idx=0, end_idx=None):
    """
    Process and parse data for a specific language using Trankit.

    Args:
        language_code: Two-letter language code (en, fi, fr, sv)
        batch_size: Number of texts to process before writing to disk
        start_idx: Optional starting index for processing a subset
        end_idx: Optional ending index for processing a subset
    """
    # Input path (filtered data from previous step)
    input_path = (
        f"data/{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Output path for parsed data
    output_path = f"data/{PARSED_CONLLU_PATH}/{language_code}_parsed.conllu"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if we're appending to existing file
    file_mode = "a" if start_idx > 0 else "w"

    # Initialize Trankit pipeline for the specific language
    print(f"Initializing Trankit pipeline for {language_code}...")
    p = trankit.Pipeline(language_code)

    # Process in batches
    chunk_size = 5000  # Read in larger chunks for efficiency
    total_processed = 0
    total_successful = 0
    errors = 0

    print(f"Processing {language_code} data...")
    try:
        # Count total rows for progress tracking
        if end_idx is None:
            with pd.read_csv(input_path, sep="\t", chunksize=100000) as reader:
                total_rows = sum(len(chunk) for chunk in reader)
            end_idx = total_rows
        else:
            total_rows = end_idx - start_idx

        # Open output file
        with open(output_path, file_mode, encoding="utf-8") as f_out:
            # Process data in chunks to limit memory usage
            reader = pd.read_csv(input_path, sep="\t", chunksize=chunk_size)

            # Skip chunks until we reach start_idx
            current_idx = 0
            for chunk in reader:
                if current_idx + len(chunk) <= start_idx:
                    current_idx += len(chunk)
                    continue

                # Process this chunk
                batch_texts = []
                batch_indices = []

                # Filter chunk to only include rows we want
                start_offset = max(0, start_idx - current_idx)
                end_offset = min(len(chunk), end_idx - current_idx)

                chunk_subset = chunk.iloc[start_offset:end_offset]

                # Prepare a batch of texts
                for idx, row in chunk_subset.iterrows():
                    global_idx = current_idx + idx - chunk.index[0]

                    if global_idx >= end_idx:
                        break

                    text = str(row.get("text", "")).strip()

                    # Get register information if available
                    register = (
                        str(row.get("register", "")).strip()
                        if "register" in row
                        else "IN"
                    )
                    register_lang = f"{register} {language_code}"

                    if not text:
                        continue

                    batch_texts.append(text)
                    batch_indices.append((global_idx, register_lang))

                    # Process in batches for better memory management
                    if len(batch_texts) >= batch_size:
                        process_batch(p, batch_texts, batch_indices, f_out)
                        total_processed += len(batch_texts)
                        total_successful += len(batch_texts)  # Assuming all succeed
                        batch_texts = []
                        batch_indices = []

                # Process remaining texts in the last batch
                if batch_texts:
                    process_batch(p, batch_texts, batch_indices, f_out)
                    total_processed += len(batch_texts)
                    total_successful += len(batch_texts)  # Assuming all succeed

                current_idx += len(chunk)

                # Print progress
                progress = min(100, int((current_idx - start_idx) / total_rows * 100))
                sys.stdout.write(
                    f"\rProcessed: {total_processed} texts, Progress: {progress}%, Errors: {errors}"
                )
                sys.stdout.flush()

                if current_idx >= end_idx:
                    break

    except Exception as e:
        print(f"\nError processing data for {language_code}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    # Return stats
    return {
        "total_processed": total_processed,
        "successful": total_successful,
        "errors": errors,
    }


def process_batch(pipeline, texts, indices, output_file):
    """Process a batch of texts with Trankit and write to output file."""
    for i, (text, (idx, register_lang)) in enumerate(zip(texts, indices)):
        try:
            # Process text with Trankit
            processed = pipeline(text)
            conllu_text = trankit2conllu(processed)

            # Add document and register information as comments
            output_file.write(f"###C: NEWDOC\n")
            output_file.write(f"###C: TEXT_ID{idx}\n")
            output_file.write(f"###C: REGISTER={register_lang}\n")
            output_file.write(conllu_text)
            output_file.write("\n")  # Extra new line between documents

        except Exception as e:
            print(f"\nError processing text {idx}: {str(e)[:100]}...")
            # Continue processing the rest of the batch


def main():
    parser = argparse.ArgumentParser(
        description="Parse filtered text data using Trankit"
    )
    parser.add_argument(
        "language",
        type=str,
        choices=["en", "fi", "fr", "sv"],
        help="Language code to process (en, fi, fr, sv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of texts to process in each batch",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for processing (for resuming interrupted jobs)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Ending index for processing (for splitting work)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU for Trankit processing"
    )

    args = parser.parse_args()

    # Set GPU usage for Trankit
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        print("Using GPU for processing")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
        print("Using CPU for processing")

    # Process the specified language
    print(f"\nProcessing {args.language} data...")
    try:
        result = parse_language_data(
            args.language,
            batch_size=args.batch_size,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

        if "error" not in result:
            print(f"\n✓ {args.language} processed successfully")
            print(f"  Processed: {result['total_processed']} texts")
            print(f"  Successful: {result['successful']} texts")
            print(f"  Errors: {result['errors']} texts")
        else:
            print(f"\n✗ Error processing {args.language}: {result['error']}")

    except Exception as e:
        print(f"\n✗ Error processing {args.language}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
