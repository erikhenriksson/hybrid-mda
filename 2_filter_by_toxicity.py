import os

# Set ALL possible cache directories FIRST - before any imports
cache_dir = "/scratch/project_2002026/ehenriks/.cache"
os.environ["TORCH_HOME"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = cache_dir
os.environ["PYTORCH_PRETRAINED_BERT_CACHE"] = cache_dir

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Also change HOME temporarily to avoid any fallback to user directory
original_home = os.environ.get("HOME")
os.environ["HOME"] = "/scratch/project_2002026/ehenriks"

import ast  # Safer alternative to eval()
import time

import pandas as pd
import torch
from detoxify import Detoxify
from transformers import AutoTokenizer

from config import FILTERED_BY_MIN_LENGTH_PATH, FILTERED_BY_TOXICITY_PATH

print("Loading models and functions...")


def read_hplt_file(file_path, sep, chunk_size):
    """Read TSV file in chunks"""
    for chunk in pd.read_csv(file_path, sep=sep, chunksize=chunk_size):
        yield chunk


def fix_label(preds_in_list_from_chunk):
    """Convert register labels to standardized format"""
    subregs = [
        "RE",
        "DTP",
        "EN",
        "FI",
        "LT",
        "RA",
        "NB",
        "NE",
        "SR",
        "DS",
        "ED",
        "AV",
        "OB",
        "RS",
        "RV",
        "IT",
    ]

    fixed_labels = []

    for pred in preds_in_list_from_chunk:
        # Remove line breaks before parsing
        pred_clean = pred.replace("\n", "").replace("\r", "")
        # Parse string representation of list to actual list - using ast.literal_eval for safety
        try:
            pred_list = ast.literal_eval(pred_clean)
        except (ValueError, SyntaxError):
            # If parsing fails, keep original value
            pred_list = pred_clean

        # Convert subregs to lowercase, keep others as-is
        if isinstance(pred_list, list):
            processed_parts = [
                part.lower() if part in subregs else part for part in pred_list
            ]
            # Convert back to string representation of list
            fixed_labels.append(str(processed_parts))
        else:
            fixed_labels.append(str(pred_list))

    return fixed_labels


def has_keywords(text, keywords):
    """Check if text contains any keywords (case-insensitive)"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def get_max_toxicity(text, detoxify_model, tokenizer):
    """Get maximum toxicity score across all text chunks"""
    tokens = tokenizer.tokenize(text, max_length=100000, truncation=True)
    if not tokens:
        return 0.0

    # Split into 512-token chunks
    token_chunks = [tokens[i : i + 512] for i in range(0, len(tokens), 512)]
    text_chunks = [tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks]

    if not text_chunks:
        return 0.0

    toxic_scores = detoxify_model.predict(text_chunks)
    return max(max(scores) for scores in toxic_scores.values())


def save_chunk_data(data, output_path, is_first_chunk):
    """Save chunk data to file with proper formatting"""
    if data:
        chunk_df = pd.DataFrame(data)
        # Remove line breaks from all text columns
        for col in chunk_df.select_dtypes(include=[object]).columns:
            chunk_df[col] = (
                chunk_df[col].astype(str).str.replace("\n", " ").str.replace("\r", " ")
            )

        # Write header only for first chunk, append for subsequent chunks
        chunk_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if is_first_chunk else "a",
            header=is_first_chunk,
        )
        return len(data)
    return 0


def process_file(
    file_path,
    clean_output_path,
    toxic_output_path,
    detoxify_model,
    tokenizer,
    use_keywords=False,
):
    """Process a single file with toxicity filtering and save both clean and toxic texts"""
    print(f"Processing {file_path}...")

    # Swedish keyword list (only used for Swedish files)
    keywords = [
        "knull",
        "fitta",
        "fittor",
        "kuk",
        "penis",
        "tuttar",
        "fuck",
        "hora",
        "horor",
        "slida",
        "porr",
        "runka",
        "sperm",
    ]

    sep = "\t"
    chunk_size = 10000
    total_processed = 0
    filtered_count = 0
    clean_count = 0

    # Track first chunk for both files
    first_clean_chunk = True
    first_toxic_chunk = True

    for data_chunk in read_hplt_file(file_path, sep, chunk_size):
        # Clean line breaks from preds column
        data_chunk["preds"] = (
            data_chunk["preds"].str.replace("\n", "").str.replace("\r", "")
        )

        # Process register labels
        data_chunk["fixed_register"] = fix_label(data_chunk.preds.to_list())

        # Remove machine translated texts
        no_mt_chunk = data_chunk[~data_chunk["preds"].str.contains("MT", na=False)]

        chunk_clean_data = []
        chunk_toxic_data = []

        for i, row in no_mt_chunk.iterrows():
            text = row["text"]
            max_toxicity = get_max_toxicity(text, detoxify_model, tokenizer)

            should_filter = False
            filter_reason = ""

            # Apply filtering logic based on language
            if max_toxicity > 0.5:
                should_filter = True
                filter_reason = "toxicity"
            elif use_keywords and has_keywords(text, keywords):
                should_filter = True
                filter_reason = "keywords"

            # Add filter reason to row for toxic file
            row_with_reason = row.copy()
            row_with_reason["filter_reason"] = (
                filter_reason if should_filter else "clean"
            )
            row_with_reason["toxicity_score"] = max_toxicity

            if not should_filter:
                # Keep this text - add to clean data
                chunk_clean_data.append(row)
                clean_count += 1
            else:
                # Filtered text - add to toxic data
                chunk_toxic_data.append(row_with_reason)
                filtered_count += 1

            total_processed += 1

            if total_processed % 100 == 0:
                print(
                    f"Processed {total_processed} texts, filtered {filtered_count}, clean {clean_count}"
                )

        # Save clean data
        saved_clean = save_chunk_data(
            chunk_clean_data, clean_output_path, first_clean_chunk
        )
        if saved_clean > 0:
            first_clean_chunk = False
            print(f"Saved {saved_clean} clean texts from current chunk")

        # Save toxic data
        saved_toxic = save_chunk_data(
            chunk_toxic_data, toxic_output_path, first_toxic_chunk
        )
        if saved_toxic > 0:
            first_toxic_chunk = False
            print(f"Saved {saved_toxic} toxic texts from current chunk")

    print(
        f"Total processed: {total_processed}, Filtered: {filtered_count}, Clean: {clean_count}"
    )
    print(f"Clean texts saved to: {clean_output_path}")
    print(f"Toxic texts saved to: {toxic_output_path}")
    return clean_count, filtered_count


# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
print("Loading models...")
detoxify_model = Detoxify("multilingual", device=device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Set up paths
input_dir = FILTERED_BY_MIN_LENGTH_PATH
output_dir = FILTERED_BY_TOXICITY_PATH

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Files to process
files_to_process = [
    ("sv_embeds_filtered.tsv", True),  # (filename, use_keywords)
    ("fr_embeds_filtered.tsv", False),
]

print("Starting toxicity filtering...")
start_time = time.time()

# Process each file
for filename, use_keywords in files_to_process:
    input_path = os.path.join(input_dir, filename)

    # Extract language code from filename
    lang_code = filename.split("_")[0]

    # Create output paths for both clean and toxic files
    clean_output_path = os.path.join(output_dir, f"{lang_code}_embeds_clean.tsv")
    toxic_output_path = os.path.join(output_dir, f"{lang_code}_embeds_toxic.tsv")

    if os.path.exists(input_path):
        language = "Swedish" if filename.startswith("sv") else "French"
        filter_type = "toxicity + keywords" if use_keywords else "toxicity only"
        print(f"\n--- Processing {language} file with {filter_type} ---")

        clean_count, filtered_count = process_file(
            input_path,
            clean_output_path,
            toxic_output_path,
            detoxify_model,
            tokenizer,
            use_keywords,
        )

        print(
            f"{language} results: {clean_count} clean texts saved, {filtered_count} filtered texts saved"
        )
    else:
        print(f"Warning: {input_path} not found, skipping...")

end_time = time.time()
total_time = end_time - start_time

print(f"\n--- FINAL RESULTS ---")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Output files saved to: {output_dir}")
print("Clean files: *_embeds_clean.tsv")
print("Toxic files: *_embeds_toxic.tsv")
print("Done!")
