import argparse
import os
import sys
import time

import pandas as pd
import trankit
from tqdm import tqdm
from trankit import trankit2conllu

from config import (
    FILTERED_BY_MEDIAN_AND_STD_PATH,
    PARSED_CONLLU_PATH,
)

# Language code mapping for Trankit
LANGUAGE_MAP = {"en": "english", "fi": "finnish", "fr": "french", "sv": "swedish"}


def parse_language_data(language_code, batch_size=1000, start_idx=0, end_idx=None):
    """
    Process and parse data for a specific language using Trankit.

    Args:
        language_code: Two-letter language code (en, fi, fr, sv)
        batch_size: Number of texts to process before writing to disk
        start_idx: Optional starting index for processing a subset
        end_idx: Optional ending index for processing a subset
    """
    # Map language code to Trankit language name
    trankit_language = LANGUAGE_MAP.get(language_code, "english")

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
    print(f"Initializing Trankit pipeline for {trankit_language}...")
    p = trankit.Pipeline(trankit_language)
    print(f"Pipeline initialized successfully.")

    # Process in batches
    chunk_size = 5000  # Read in larger chunks for efficiency
    total_processed = 0
    total_successful = 0
    errors = 0
    start_time = time.time()
    last_time = start_time

    print(f"Processing {language_code} data from file: {input_path}")
    print(f"Output will be written to: {output_path}")
    print(f"Processing mode: {'Append' if file_mode == 'a' else 'Create new'}")

    try:
        # Count total rows for progress tracking if end_idx not specified
        if end_idx is None:
            print(
                "Counting total rows in file (this may take a while for large files)..."
            )
            count_start = time.time()
            with pd.read_csv(input_path, sep="\t", chunksize=100000) as reader:
                total_rows = 0
                for chunk in reader:
                    total_rows += len(chunk)
                    # Print count progress every million rows
                    if total_rows % 1000000 == 0:
                        print(f"  Counted {total_rows:,} rows...")
            end_idx = total_rows
            print(
                f"Finished counting. Total rows: {total_rows:,} (took {time.time() - count_start:.2f} seconds)"
            )
        else:
            total_rows = end_idx - start_idx
            print(
                f"Will process rows from {start_idx:,} to {end_idx:,} (total: {total_rows:,} rows)"
            )

        # Open output file
        with open(output_path, file_mode, encoding="utf-8") as f_out:
            # Process data in chunks to limit memory usage
            print(
                f"Starting to read and process data in chunks of {chunk_size:,} rows..."
            )
            reader = pd.read_csv(input_path, sep="\t", chunksize=chunk_size)

            # Skip chunks until we reach start_idx
            current_idx = 0
            chunk_count = 0

            for chunk in reader:
                chunk_count += 1

                # Skip chunks before our starting point
                if current_idx + len(chunk) <= start_idx:
                    current_idx += len(chunk)
                    print(
                        f"Skipping chunk {chunk_count} (rows {current_idx - len(chunk):,} to {current_idx:,})..."
                    )
                    continue

                # Process this chunk
                chunk_start_time = time.time()
                print(
                    f"Processing chunk {chunk_count} (rows {current_idx:,} to {current_idx + len(chunk):,})..."
                )

                batch_texts = []
                batch_indices = []

                # Filter chunk to only include rows we want
                start_offset = max(0, start_idx - current_idx)
                end_offset = min(len(chunk), end_idx - current_idx)

                chunk_subset = chunk.iloc[start_offset:end_offset]
                print(f"  Subset to process: {len(chunk_subset):,} rows")

                # Track batch progress within chunk
                batch_count = 0

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
                        batch_count += 1
                        batch_start = time.time()
                        print(
                            f"  Processing batch {batch_count} ({len(batch_texts):,} texts)..."
                        )
                        process_batch(p, batch_texts, batch_indices, f_out)

                        batch_successful = len(batch_texts)
                        total_processed += batch_successful
                        total_successful += batch_successful

                        batch_time = time.time() - batch_start
                        texts_per_second = batch_successful / batch_time

                        print(
                            f"  Batch completed in {batch_time:.2f} seconds ({texts_per_second:.2f} texts/second)"
                        )
                        batch_texts = []
                        batch_indices = []

                # Process remaining texts in the last batch
                if batch_texts:
                    batch_count += 1
                    print(
                        f"  Processing final batch {batch_count} ({len(batch_texts):,} texts)..."
                    )
                    batch_start = time.time()

                    process_batch(p, batch_texts, batch_indices, f_out)

                    batch_successful = len(batch_texts)
                    total_processed += batch_successful
                    total_successful += batch_successful

                    batch_time = time.time() - batch_start
                    texts_per_second = (
                        batch_successful / batch_time if batch_time > 0 else 0
                    )

                    print(
                        f"  Batch completed in {batch_time:.2f} seconds ({texts_per_second:.2f} texts/second)"
                    )

                current_idx += len(chunk)
                chunk_time = time.time() - chunk_start_time

                # Calculate progress and speeds
                progress = min(100, ((current_idx - start_idx) / total_rows) * 100)
                elapsed_time = time.time() - start_time

                # Calculate processing speed
                texts_per_second = (
                    total_processed / elapsed_time if elapsed_time > 0 else 0
                )

                # Calculate estimated time remaining
                if progress > 0:
                    estimated_total_time = elapsed_time / (progress / 100)
                    estimated_remaining = estimated_total_time - elapsed_time

                    # Format time remaining as hours:minutes:seconds
                    hours, remainder = divmod(estimated_remaining, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_remaining_str = (
                        f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    )
                else:
                    time_remaining_str = "calculating..."

                # Print comprehensive progress information
                print(f"Chunk completed in {chunk_time:.2f} seconds")
                print(
                    f"Overall progress: {progress:.2f}% ({total_processed:,}/{total_rows:,} texts)"
                )
                print(f"Processing speed: {texts_per_second:.2f} texts/second")
                print(f"Estimated time remaining: {time_remaining_str}")
                print(f"Errors: {errors}")
                print("-" * 50)

                # Exit if we've reached our end index
                if current_idx >= end_idx:
                    print(
                        f"Reached target end index ({end_idx:,}), stopping processing."
                    )
                    break

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return {
            "total_processed": total_processed,
            "successful": total_successful,
            "errors": errors,
            "interrupted": True,
        }
    except Exception as e:
        print(f"\nError processing data for {language_code}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    # Calculate total elapsed time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Return stats
    return {
        "total_processed": total_processed,
        "successful": total_successful,
        "errors": errors,
        "processing_time": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
        "texts_per_second": total_processed / total_time if total_time > 0 else 0,
    }


def process_batch(pipeline, texts, indices, output_file):
    """Process a batch of texts with Trankit and write to output file."""
    batch_errors = 0

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
            batch_errors += 1
            print(f"Error processing text {idx}: {str(e)[:100]}...")
            # Continue processing the rest of the batch

    return batch_errors


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
        default=100,
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
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only a sample of N texts (for testing)",
    )

    args = parser.parse_args()

    # If sample is specified, set end_idx accordingly
    if args.sample is not None:
        args.end_idx = args.start_idx + args.sample
        print(
            f"Sample mode: Will process {args.sample} texts starting from index {args.start_idx}"
        )

    # Validate language
    if args.language not in LANGUAGE_MAP:
        print(f"Error: Unsupported language code '{args.language}'")
        print(f"Supported languages: {', '.join(LANGUAGE_MAP.keys())}")
        sys.exit(1)

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
            print("\n" + "=" * 50)
            print(f"Processing Summary for {args.language}:")
            print("=" * 50)
            print(f"Total texts processed: {result['total_processed']:,}")
            print(f"Successfully processed: {result['successful']:,}")
            print(f"Errors encountered: {result['errors']:,}")

            if "processing_time" in result:
                print(f"Total processing time: {result['processing_time']}")
                print(
                    f"Average processing speed: {result['texts_per_second']:.2f} texts/second"
                )

            if result.get("interrupted", False):
                print(f"\nNote: Processing was interrupted. To resume, use:")
                print(
                    f"python3 {sys.argv[0]} {args.language} --start_idx {args.start_idx + result['total_processed']} --gpu"
                )

            print("=" * 50)
        else:
            print(f"\n✗ Error processing {args.language}: {result['error']}")

    except Exception as e:
        print(f"\n✗ Error processing {args.language}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
