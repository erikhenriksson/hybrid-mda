import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd
import torch
import trankit
from tqdm import tqdm
from trankit import trankit2conllu

from config import FILTERED_BY_MEDIAN_AND_STD_PATH, PARSED_CONLLU_PATH


def process_chunk_gpu(chunk_data, language, gpu_id, chunk_idx, total_chunks):
    """Process a chunk of data with trankit on a specific GPU."""
    try:
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Initialize trankit pipeline with GPU
        p = trankit.Pipeline(language, gpu=True)

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
                print(f"GPU {gpu_id}: Error processing row {idx}: {str(e)}")
                continue

        # Write chunk results to file
        chunk_filename = (
            f"gpu_{gpu_id}_chunk_{chunk_idx:04d}_of_{total_chunks:04d}.conllu"
        )
        chunk_path = os.path.join(PARSED_CONLLU_PATH, "temp_chunks", chunk_filename)

        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        with open(chunk_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result)

        print(
            f"GPU {gpu_id}: Completed chunk {chunk_idx}/{total_chunks} with {len(results)} documents"
        )
        return chunk_path, len(results)

    except Exception as e:
        print(f"GPU {gpu_id}: Error processing chunk {chunk_idx}: {str(e)}")
        return None, 0


def parse_language_data_gpu(language_code, n_gpus=8):
    """Parse data for a specific language using multiple GPUs."""
    print(f"Using {n_gpus} GPUs for parallel processing")

    # Check GPU availability
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. Please ensure you're running on a GPU node."
        )
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if available_gpus < n_gpus:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead of {n_gpus}"
        )
        n_gpus = available_gpus

    # Input path
    input_path = (
        f"data/{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Output path
    output_path = f"data/{PARSED_CONLLU_PATH}/{language_code}_parsed.conllu"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process in chunks optimized for GPU processing
    chunk_size = 10000  # Smaller chunks for GPU processing

    print(f"Reading input file: {input_path}")
    print("This might take a while for large files...")

    # First pass: count chunks
    total_chunks = 0
    for i, _ in enumerate(pd.read_csv(input_path, sep="\t", chunksize=chunk_size)):
        total_chunks = i + 1

    print(f"File has {total_chunks} chunks of {chunk_size} rows each")

    # Process chunks with GPU assignment
    chunk_paths = []
    processed_rows = 0

    # Create pool of GPU workers
    with Pool(processes=n_gpus) as pool:
        # Prepare chunks with GPU assignments
        tasks = []
        for idx, chunk in enumerate(
            pd.read_csv(input_path, sep="\t", chunksize=chunk_size)
        ):
            gpu_id = idx % n_gpus  # Round-robin GPU assignment
            tasks.append((chunk, language_code, gpu_id, idx, total_chunks))

        # Process chunks in parallel
        results = []
        for task in tqdm(tasks, desc="Processing chunks on GPUs", total=len(tasks)):
            result = pool.apply_async(process_chunk_gpu, task)
            results.append(result)

        # Collect results
        for result in tqdm(results, desc="Collecting results", total=len(results)):
            try:
                chunk_path, rows_processed = result.get(
                    timeout=600
                )  # 10 minute timeout per chunk
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
        description="Parse text data using Trankit with GPU acceleration"
    )
    parser.add_argument(
        "language", help="Language code to process (e.g., en, fi, fr, sv)"
    )
    parser.add_argument(
        "--gpus", type=int, default=8, help="Number of GPUs to use (default: 8)"
    )

    args = parser.parse_args()

    # Validate language
    supported_languages = ["en", "fi", "fr", "sv"]
    if args.language not in supported_languages:
        print(f"Error: Language '{args.language}' not supported.")
        print(f"Supported languages: {', '.join(supported_languages)}")
        sys.exit(1)

    print(f"\nStarting GPU-accelerated parsing for language: {args.language}")

    try:
        results = parse_language_data_gpu(args.language, args.gpus)

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
