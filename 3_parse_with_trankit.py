import argparse
import gc  # Garbage collection for memory management
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
import torch
import trankit
from trankit import trankit2conllu

from config import FILTERED_BY_MEDIAN_AND_STD_PATH, PARSED_CONLLU_PATH

# Define language mapping for Trankit
# Based on supported languages in Trankit
TRANKIT_LANGUAGE_MAP = {
    "en": "english",
    "fr": "french",
    "fi": "finnish",
    "sv": "swedish",  # Try this, or we might need to use a different model
}


def process_chunk_gpu(task_data):
    """Process a chunk of data with trankit on a specific GPU."""
    chunk_data, language, gpu_id, chunk_idx = task_data

    try:
        start_time = time.time()
        # Set CUDA device for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Get the proper language name for Trankit
        trankit_language = TRANKIT_LANGUAGE_MAP.get(language, language)

        # Initialize trankit pipeline with GPU
        try:
            p = trankit.Pipeline(trankit_language, gpu=True)
        except Exception as e:
            print(
                f"Error initializing Trankit pipeline with language '{trankit_language}': {str(e)}"
            )
            print(f"Supported languages in Trankit: {trankit.supported_langs}")
            print("Attempting to use 'english' as fallback...")
            p = trankit.Pipeline("english", gpu=True)

        # Initialize the results list
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
                    f"###C: REGISTER={register} in {language}",
                ]

                result_text = "\n".join(headers) + "\n" + conllu_text + "\n"
                results.append(result_text)

            except Exception as e:
                # Log error but continue processing
                print(f"GPU {gpu_id}: Error processing row {idx}: {str(e)}")
                continue

        # Write chunk results to file
        chunk_filename = f"gpu_{gpu_id}_chunk_{chunk_idx:04d}.conllu"
        chunk_path = os.path.join(PARSED_CONLLU_PATH, "temp_chunks", chunk_filename)

        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        with open(chunk_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result)

        elapsed_time = time.time() - start_time
        docs_per_second = len(results) / elapsed_time if elapsed_time > 0 else 0

        print(
            f"GPU {gpu_id}: Completed chunk {chunk_idx} with {len(results)} documents in {elapsed_time:.2f}s ({docs_per_second:.2f} docs/s)"
        )

        # Clean up memory
        del chunk_data, results, p
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU memory

        return chunk_path, len(results)

    except Exception as e:
        print(f"GPU {gpu_id}: Error processing chunk {chunk_idx}: {str(e)}")
        print(f"Exception details: {str(e)}")
        import traceback

        traceback.print_exc()
        # Return empty results if exception occurred
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

    # Print supported languages from Trankit
    print(f"Supported languages in Trankit: {trankit.supported_langs}")

    # Input path
    input_path = (
        f"data/{FILTERED_BY_MEDIAN_AND_STD_PATH}/{language_code}_embeds_filtered.tsv"
    )

    # Output path
    output_path = f"data/{PARSED_CONLLU_PATH}/{language_code}_parsed.conllu"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.join(PARSED_CONLLU_PATH, "temp_chunks"), exist_ok=True)

    # Process in chunks optimized for GPU processing
    chunk_size = 10000  # Smaller chunks for GPU processing

    print(f"Starting processing of: {input_path}")
    print(f"Processing with chunk size: {chunk_size}")
    print("Processing will begin immediately without counting total chunks first")

    # Process chunks with GPU assignment
    chunk_paths = []
    processed_rows = 0
    chunk_count = 0

    start_time = time.time()

    with Pool(processes=n_gpus) as pool:
        # Set up tasks for processing
        active_tasks = []
        max_active_tasks = n_gpus * 2  # Allow 2 tasks per GPU in queue

        # Read and process chunks
        for chunk_idx, chunk in enumerate(
            pd.read_csv(input_path, sep="\t", chunksize=chunk_size)
        ):
            gpu_id = chunk_idx % n_gpus  # Round-robin GPU assignment

            # Create and submit task
            task = (chunk, language_code, gpu_id, chunk_idx)
            result = pool.apply_async(process_chunk_gpu, (task,))
            active_tasks.append((result, chunk_idx))

            # Check completed tasks
            i = 0
            while i < len(active_tasks):
                result, idx = active_tasks[i]
                if result.ready():
                    try:
                        chunk_path, rows = result.get(timeout=1)
                        if chunk_path:
                            chunk_paths.append(chunk_path)
                            processed_rows += rows
                            chunk_count += 1

                        # Remove from active tasks
                        active_tasks.pop(i)

                        # Show periodic progress
                        if chunk_count % 10 == 0:
                            elapsed = time.time() - start_time
                            docs_per_second = (
                                processed_rows / elapsed if elapsed > 0 else 0
                            )
                            print(
                                f"Progress: {chunk_count} chunks processed, {processed_rows} documents, {docs_per_second:.2f} docs/s"
                            )
                    except Exception as e:
                        print(f"Error getting result for chunk {idx}: {str(e)}")
                        active_tasks.pop(i)
                else:
                    i += 1

            # Wait if too many active tasks
            while len(active_tasks) >= max_active_tasks:
                # Sleep briefly
                time.sleep(0.5)

                # Check completed tasks
                i = 0
                while i < len(active_tasks):
                    result, idx = active_tasks[i]
                    if result.ready():
                        try:
                            chunk_path, rows = result.get(timeout=1)
                            if chunk_path:
                                chunk_paths.append(chunk_path)
                                processed_rows += rows
                                chunk_count += 1

                            # Remove from active tasks
                            active_tasks.pop(i)

                            # Show periodic progress
                            if chunk_count % 10 == 0:
                                elapsed = time.time() - start_time
                                docs_per_second = (
                                    processed_rows / elapsed if elapsed > 0 else 0
                                )
                                print(
                                    f"Progress: {chunk_count} chunks processed, {processed_rows} documents, {docs_per_second:.2f} docs/s"
                                )
                        except Exception as e:
                            print(f"Error getting result for chunk {idx}: {str(e)}")
                            active_tasks.pop(i)
                    else:
                        i += 1

        # Wait for remaining tasks to complete
        print("All chunks submitted, waiting for remaining tasks to complete...")
        for result, idx in active_tasks:
            try:
                chunk_path, rows = result.get(timeout=300)  # 5 minute timeout
                if chunk_path:
                    chunk_paths.append(chunk_path)
                    processed_rows += rows
                    chunk_count += 1
            except Exception as e:
                print(f"Error getting result for chunk {idx}: {str(e)}")

    # Final progress report
    elapsed = time.time() - start_time
    docs_per_second = processed_rows / elapsed if elapsed > 0 else 0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    print(f"Processing finished! Total time: {hours}h {minutes}m {seconds}s")
    print(f"Processed {chunk_count} chunks, {processed_rows} documents")
    print(f"Average processing speed: {docs_per_second:.2f} docs/s")

    # Merge chunk files
    print(f"\nMerging {len(chunk_paths)} chunk files...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, chunk_path in enumerate(sorted(chunk_paths)):
            try:
                print(
                    f"Merging file {i + 1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}"
                )
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
