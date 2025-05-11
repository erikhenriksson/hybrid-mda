import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
import trankit
from tqdm import tqdm
from trankit import trankit2conllu

from config import FILTERED_BY_MEDIAN_AND_STD_PATH, PARSED_CONLLU_PATH


def process_chunk(chunk_data, language, chunk_idx, total_chunks):
    """Process a chunk of data with trankit."""
    try:
        # Initialize trankit pipeline for this process
        p = trankit.Pipeline(language)

        results = []
        for idx, row in chunk_data.iterrows():
            try:
                # Extract text and other columns
                text = str(row["text"]).strip()
                if not text:
                    continue

                # Get register from 'preds' or use a default
                register = str(row.get("preds", "UNK"))

                # Process with trankit
                processed = p(text)
                conllu_text = trankit2conllu(processed)

                # Create document with headers
                doc_id = f"TEXT_ID{idx}"
                headers = [
                    "###C: NEWDOC",
                    f"###C: {doc_id}",
                    f"###C: REGISTER={register}",
                ]

                result_text = "\n".join(headers) + "\n" + conllu_text + "\n"
                results.append(result_text)

            except Exception as e:
                # Log error but continue processing
                print(f"Error processing row {idx}: {str(e)}")
                continue

        # Write chunk results to file
        chunk_filename = f"chunk_{chunk_idx:04d}_of_{total_chunks:04d}.conllu"
        chunk_path = os.path.join(PARSED_CONLLU_PATH, "temp_chunks", chunk_filename)

        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        with open(chunk_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result)

        return chunk_path, len(results)

    except Exception as e:
        print(f"Error processing chunk {chunk_idx}: {str(e)}")
        return None, 0


def parse_language_data(language_code, n_processes=None):
    """Parse data for a specific language using multiprocessing."""
    # Set number of processes
    if n_processes is None:
        n_processes = min(cpu_count(), 16)  # Cap at 16 to avoid overwhelming trankit

    print(f"Using {n_processes} processes for parallel processing")

    # Input path
    input_path = (
        f"data/{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Output path
    output_path = f"data/{PARSED_CONLLU_PATH}/{language_code}_parsed.conllu"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process in large chunks for multiprocessing
    chunk_size = 50000  # Larger chunks for multiprocessing

    print(f"Reading input file: {input_path}")
    print("This might take a while for large files...")

    # Create partial function with language and total chunks info
    total_chunks = 0

    # First pass: count chunks
    for i, _ in enumerate(pd.read_csv(input_path, sep="\t", chunksize=chunk_size)):
        total_chunks = i + 1

    print(f"File has {total_chunks} chunks of {chunk_size} rows each")

    # Process chunks in parallel
    chunk_paths = []
    processed_rows = 0

    # Create a partial function with fixed arguments
    process_func = partial(
        process_chunk, language=language_code, total_chunks=total_chunks
    )

    with Pool(processes=n_processes) as pool:
        # Create iterator over chunks with indices
        chunks_with_idx = []
        for idx, chunk in enumerate(
            pd.read_csv(input_path, sep="\t", chunksize=chunk_size)
        ):
            chunks_with_idx.append((chunk, idx))

        # Process chunks in parallel
        results = []
        for chunk, chunk_idx in tqdm(
            chunks_with_idx, desc="Processing chunks", total=total_chunks
        ):
            result = pool.apply_async(process_func, (chunk, chunk_idx))
            results.append(result)

        # Collect results
        for result in tqdm(results, desc="Collecting results", total=len(results)):
            try:
                chunk_path, rows_processed = result.get(
                    timeout=300
                )  # 5 minute timeout per chunk
                if chunk_path:
                    chunk_paths.append(chunk_path)
                    processed_rows += rows_processed
            except Exception as e:
                print(f"Error collecting result: {str(e)}")
                continue

    # Merge chunk files
    print(f"\nMerging {len(chunk_paths)} chunk files...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, chunk_path in enumerate(tqdm(chunk_paths, desc="Merging chunks")):
            try:
                with open(chunk_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()
                    f_out.write(content)
                    if i < len(chunk_paths) - 1:  # Add double newline between documents
                        f_out.write("\n")
                # Remove temporary chunk file
                os.remove(chunk_path)
            except Exception as e:
                print(f"Error reading chunk file {chunk_path}: {str(e)}")
                continue

    # Clean up temp directory
    temp_dir = os.path.join(PARSED_CONLLU_PATH, "temp_chunks")
    try:
        os.rmdir(temp_dir)
    except:
        pass

    print(f"\nProcessing complete!")
    print(f"Processed {processed_rows} rows")
    print(f"Output saved to: {output_path}")

    return {
        "language": language_code,
        "processed_rows": processed_rows,
        "output_file": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parse text data using Trankit to CoNLL-U format"
    )
    parser.add_argument(
        "language", help="Language code to process (e.g., en, fi, fr, sv)"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes to use (default: auto-detect, max 16)",
    )

    args = parser.parse_args()

    # Validate language
    supported_languages = ["en", "fi", "fr", "sv"]
    if args.language not in supported_languages:
        print(f"Error: Language '{args.language}' not supported.")
        print(f"Supported languages: {', '.join(supported_languages)}")
        sys.exit(1)

    print(f"\nStarting parsing for language: {args.language}")

    try:
        results = parse_language_data(args.language, args.processes)

        print("\nParsing Summary:")
        print("================")
        print(f"Language: {results['language']}")
        print(f"Processed rows: {results['processed_rows']}")
        print(f"Output file: {results['output_file']}")

    except Exception as e:
        print(f"\nError: Failed to process {args.language}")
        print(f"Error details: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
