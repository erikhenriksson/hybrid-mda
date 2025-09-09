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
        # Parse string representation of list to actual list
        pred_list = eval(pred_clean)

        # Convert subregs to lowercase, keep others as-is
        processed_parts = [
            part.lower() if part in subregs else part for part in pred_list
        ]
        # Convert back to string representation of list
        fixed_labels.append(str(processed_parts))

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


def process_file(file_path, output_path, detoxify_model, tokenizer, use_keywords=False):
    """Process a single file with toxicity filtering"""
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
    clean_data = []
    total_processed = 0
    filtered_count = 0

    for data_chunk in read_hplt_file(file_path, sep, chunk_size):
        # Clean line breaks from preds column
        data_chunk["preds"] = (
            data_chunk["preds"].str.replace("\n", "").str.replace("\r", "")
        )

        # Process register labels
        data_chunk["fixed_register"] = fix_label(data_chunk.preds.to_list())

        # Remove machine translated texts
        no_mt_chunk = data_chunk[~data_chunk["preds"].str.contains("MT", na=False)]

        for i, row in no_mt_chunk.iterrows():
            text = row["text"]
            max_toxicity = get_max_toxicity(text, detoxify_model, tokenizer)

            should_filter = False

            # Apply filtering logic based on language
            if max_toxicity > 0.5:
                should_filter = True
            elif use_keywords and has_keywords(text, keywords):
                should_filter = True

            if not should_filter:
                # Keep this text - add to clean data
                clean_data.append(row)
            else:
                filtered_count += 1

            total_processed += 1

            if total_processed % 100 == 0:
                print(f"Processed {total_processed} texts, filtered {filtered_count}")

    # Save clean data to output file
    if clean_data:
        clean_df = pd.DataFrame(clean_data)
        # Remove line breaks from all text columns when saving
        for col in clean_df.select_dtypes(include=[object]).columns:
            clean_df[col] = (
                clean_df[col].astype(str).str.replace("\n", " ").str.replace("\r", " ")
            )

        clean_df.to_csv(output_path, sep="\t", index=False)
        print(f"Saved {len(clean_data)} clean texts to {output_path}")

    print(
        f"Total processed: {total_processed}, Filtered: {filtered_count}, Clean: {len(clean_data)}"
    )
    return len(clean_data), filtered_count


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
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(input_path):
        language = "Swedish" if filename.startswith("sv") else "French"
        filter_type = "toxicity + keywords" if use_keywords else "toxicity only"
        print(f"\n--- Processing {language} file with {filter_type} ---")

        clean_count, filtered_count = process_file(
            input_path, output_path, detoxify_model, tokenizer, use_keywords
        )

        print(
            f"{language} results: {clean_count} clean texts saved, {filtered_count} filtered out"
        )
    else:
        print(f"Warning: {input_path} not found, skipping...")

end_time = time.time()
total_time = end_time - start_time

print(f"\n--- FINAL RESULTS ---")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Output files saved to: {output_dir}")
print("Done!")
