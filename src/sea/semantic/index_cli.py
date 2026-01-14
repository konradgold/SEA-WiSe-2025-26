import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sentence_transformers import SentenceTransformer

from sea.storage.embeddings import EmbeddingIO
from sea.utils.config_wrapper import Config
from sea.utils.device import detect_device


MAX_TEXT_LENGTH = 2048


def read_documents_batch(tsv_path: str, start_row: int, row_count: int) -> list[tuple[int, str]]:
    """Read batch of docs from TSV with title and body
    Text is truncated to MAX_TEXT_LENGTH chars
    """
    documents = []
    end_row = start_row + row_count

    with open(tsv_path, "r", encoding="utf-8") as tsv_file:
        for row_index, line in enumerate(tsv_file):
            if row_index < start_row:
                continue
            if row_index >= end_row:
                break

            columns = line.strip().split("\t")
            title = columns[2] if len(columns) > 2 else ""
            body = columns[3] if len(columns) > 3 else ""

            # Combine title + body
            remaining_chars = MAX_TEXT_LENGTH - len(title) - 1  # -1 for space
            if remaining_chars > 0 and body:
                combined_text = f"{title} {body[:remaining_chars]}"
            else:
                combined_text = title

            # Use placeholder for blank docs
            documents.append((row_index, combined_text.strip() or "empty"))

    return documents


def compute_embeddings_batches(
    model: SentenceTransformer,
    texts: list[str],
    embedding_dimension: int,
    batch_size: int,
):
    """Yield normalized Matryoshka embeddings in batches, same normalization pipeline as the embedding:
    layer norm -> truncate -> L2 normalize.
    """
    # Add document prefix required by model
    prefixed_texts = ["search_document: " + text for text in texts]

    for batch_start in range(0, len(prefixed_texts), batch_size):
        batch_texts = prefixed_texts[batch_start : batch_start + batch_size]
        raw_embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        # Matryoshka reduction 
        normalized = F.layer_norm(raw_embeddings, normalized_shape=(raw_embeddings.shape[1],))
        truncated = normalized[:, :embedding_dimension]
        unit_vectors = F.normalize(truncated, p=2, dim=1)

        yield unit_vectors.cpu().numpy().astype(np.float32)


def get_checkpoint_path(data_path: str) -> Path:
    return Path(data_path) / "embedding_checkpoint.json"


def load_checkpoint(data_path: str) -> dict:
    path = get_checkpoint_path(data_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_docs": 0, "partial_path": None}


def save_checkpoint(data_path: str, completed_docs: int, partial_path: str):
    path = get_checkpoint_path(data_path)
    with open(path, "w") as f:
        json.dump({"completed_docs": completed_docs, "partial_path": partial_path}, f)


def main():
    parser = argparse.ArgumentParser(description="Compute document embeddings")
    parser.add_argument("--num_docs", type=int, default=-1, help="Total documents to embed (-1 for all)")
    parser.add_argument("--batch_size", type=int, help="Batch size (default from config)")
    parser.add_argument("--checkpoint_every", type=int, default=50000, help="Save checkpoint every N docs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="Overwrite existing embeddings file")
    args = parser.parse_args()

    config = Config(load=True)

    # Check for existing embeddings file
    if not args.resume:
        embeddings_path = os.path.join(config.DATA_PATH, "embeddings.bin")
        if os.path.exists(embeddings_path) and not args.force:
            print(f"Error: Embeddings file already exists: {embeddings_path}")
            print("\nUse --force to overwrite existing file, or --resume to continue from checkpoint.")
            sys.exit(1)

    model_id = config.SEMANTIC.MODEL_ID
    requested_device = config.SEMANTIC.DEVICE
    device = detect_device(requested_device)
    embedding_dimension = config.SEMANTIC.DIM
    batch_size = args.batch_size or config.SEMANTIC.BATCH_SIZE
    documents_path = config.DOCUMENTS
    data_directory = config.DATA_PATH

    # Check for checkpoint
    checkpoint = load_checkpoint(data_directory) if args.resume else {"completed_docs": 0, "partial_path": None}
    start_document_index = checkpoint["completed_docs"]

    if start_document_index > 0:
        print(f"Resuming from document {start_document_index:,}")

    print(f"Loading model {model_id} on {device}...")
    model = SentenceTransformer(model_id, trust_remote_code=True, device=device)

    # Process in chunks with checkpointing for large datasets
    documents_per_checkpoint = args.checkpoint_every
    all_embeddings = []

    # Load existing partial embeddings if resuming
    partial_embeddings_path = checkpoint.get("partial_path")
    if partial_embeddings_path and os.path.exists(partial_embeddings_path):
        print(f"Loading partial embeddings from {partial_embeddings_path}...")
        all_embeddings.append(np.load(partial_embeddings_path))

    # Count total docs if not specified
    total_documents = args.num_docs
    if total_documents < 0:
        print("Counting documents in corpus...")
        with open(documents_path, "r", encoding="utf-8") as file:
            total_documents = sum(1 for _ in file)
        print(f"Found {total_documents:,} documents")

    progress_bar = tqdm.tqdm(total=total_documents, initial=start_document_index, desc="Embedding docs")

    start_time = perf_counter()
    current_document_index = start_document_index
    next_checkpoint_at = ((current_document_index // documents_per_checkpoint) + 1) * documents_per_checkpoint

    while current_document_index < total_documents:
        # Read chunk of documents
        documents_to_read = min(documents_per_checkpoint, total_documents - current_document_index)
        documents = read_documents_batch(documents_path, current_document_index, documents_to_read)

        if not documents:
            break

        document_texts = [text for _, text in documents]

        # Embed in smaller batches so progress bar advances smoothly
        for embedding_batch in compute_embeddings_batches(model, document_texts, embedding_dimension, batch_size):
            all_embeddings.append(embedding_batch)
            current_document_index += embedding_batch.shape[0]
            progress_bar.update(embedding_batch.shape[0])

            # Save checkpoint when crossing scheduled boundaries
            if current_document_index < total_documents and current_document_index >= next_checkpoint_at:
                partial_embeddings = np.vstack(all_embeddings)
                partial_embeddings_path = os.path.join(data_directory, "embeddings_partial.npy")
                np.save(partial_embeddings_path, partial_embeddings)
                save_checkpoint(data_directory, current_document_index, partial_embeddings_path)

                elapsed_seconds = perf_counter() - start_time
                documents_per_second = current_document_index / elapsed_seconds if elapsed_seconds > 0 else 0.0
                remaining_seconds = (
                    (total_documents - current_document_index) / documents_per_second
                    if documents_per_second > 0
                    else float("inf")
                )
                print(f"\nCheckpoint at {current_document_index:,} docs. ETA: {remaining_seconds/60:.1f} min")
                next_checkpoint_at += documents_per_checkpoint

    progress_bar.close()

    # Combine and save final embeddings
    final_embeddings = np.vstack(all_embeddings)
    print(f"\nFinal shape: {final_embeddings.shape}")

    # Write to binary format
    embedding_io = EmbeddingIO(config)
    embedding_io.write(final_embeddings, force=args.force or args.resume)

    # Clean up checkpoint files
    checkpoint_file_path = get_checkpoint_path(data_directory)
    if checkpoint_file_path.exists():
        checkpoint_file_path.unlink()
    partial_embeddings_path = os.path.join(data_directory, "embeddings_partial.npy")
    if os.path.exists(partial_embeddings_path):
        os.remove(partial_embeddings_path)

    total_time_seconds = perf_counter() - start_time
    size_megabytes = final_embeddings.nbytes / (1024 * 1024)
    print(f"Done! {final_embeddings.shape[0]:,} docs, {embedding_dimension} dims, {size_megabytes:.1f} MB")
    print(f"Time: {total_time_seconds/60:.1f} min ({final_embeddings.shape[0]/total_time_seconds:.1f} docs/sec)")


if __name__ == "__main__":
    main()
