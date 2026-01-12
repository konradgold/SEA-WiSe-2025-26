"""CLI for computing and storing document embeddings.

For large corpora (3.3M docs), key optimizations:
- GPU with large batches
- Checkpointing to resume on crash
- Progress saved periodically

Usage:
    uv run python -m sea.semantic.index_cli --num_docs 3300000
    uv run python -m sea.semantic.index_cli --resume  # continue from checkpoint
"""

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


# Truncate to ~512 tokens worth of text (avg 4 chars/token)
MAX_CHARS = 2048


def detect_device(requested: str) -> str:
    """Auto-detect the best available device for PyTorch.

    Args:
        requested: Device from config ("auto", "cuda", "cuda:0", "mps", "cpu")

    Returns:
        Validated device string that is actually available.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # Validate requested device is available
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        # Check specific device index if provided (e.g., "cuda:1")
        if ":" in requested:
            device_idx = int(requested.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                print(f"Warning: {requested} not available (only {torch.cuda.device_count()} GPUs). Using cuda:0.")
                return "cuda:0"
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(f"Warning: MPS requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    return requested  # cpu or unknown


def read_documents_batch(tsv_path: str, start: int, count: int) -> list[tuple[int, str]]:
    """Read a batch of documents starting at row `start`."""
    docs = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + count:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                title = parts[2] if len(parts) > 2 else ""
                body = parts[3] if len(parts) > 3 else ""
                # Prioritize title, then as much body as fits
                text = title
                remaining = MAX_CHARS - len(title) - 1
                if remaining > 0 and body:
                    text = f"{title} {body[:remaining]}"
                docs.append((i, text.strip()))
    return docs


def compute_embeddings_batch(
    model: SentenceTransformer,
    texts: list[str],
    dim: int,
    batch_size: int,
) -> np.ndarray:
    """Compute normalized Matryoshka embeddings."""
    prefixed = ["search_document: " + t for t in texts]

    all_embeddings = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        # Use model's native encoding with truncation
        emb = model.encode(
            batch,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # We normalize after Matryoshka truncation
        )

        # Matryoshka: layer_norm -> truncate -> L2 normalize
        emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
        emb = emb[:, :dim]
        emb = F.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


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

    cfg = Config(load=True)

    # Check for existing embeddings file 
    if not args.resume:
        embeddings_path = os.path.join(cfg.DATA_PATH, "embeddings.bin")
        if os.path.exists(embeddings_path) and not args.force:
            print(f"Error: Embeddings file already exists: {embeddings_path}")
            print("\nUse --force to overwrite existing file, or --resume to continue from checkpoint.")
            sys.exit(1)
    model_id = cfg.SEMANTIC.MODEL_ID
    requested_device = cfg.SEMANTIC.DEVICE
    device = detect_device(requested_device)
    dim = cfg.SEMANTIC.DIM
    batch_size = args.batch_size or cfg.SEMANTIC.BATCH_SIZE
    docs_path = cfg.DOCUMENTS
    data_path = cfg.DATA_PATH

    # Check for checkpoint
    checkpoint = load_checkpoint(data_path) if args.resume else {"completed_docs": 0, "partial_path": None}
    start_doc = checkpoint["completed_docs"]

    if start_doc > 0:
        print(f"Resuming from document {start_doc:,}")

    print(f"Loading model {model_id} on {device}...")
    model = SentenceTransformer(model_id, trust_remote_code=True, device=device)

    # Process in chunks with checkpointing
    chunk_size = args.checkpoint_every
    all_embeddings = []

    # Load existing partial embeddings if resuming
    partial_path = checkpoint.get("partial_path")
    if partial_path and os.path.exists(partial_path):
        print(f"Loading partial embeddings from {partial_path}...")
        all_embeddings.append(np.load(partial_path))

    # Count total docs if needed
    total_docs = args.num_docs
    if total_docs < 0:
        print("Counting documents in corpus...")
        with open(docs_path, "r", encoding="utf-8") as f:
            total_docs = sum(1 for _ in f)
        print(f"Found {total_docs:,} documents")

    pbar = tqdm.tqdm(total=total_docs, initial=start_doc, desc="Embedding docs")

    t0 = perf_counter()
    current_doc = start_doc

    while current_doc < total_docs:
        # Read chunk
        chunk_count = min(chunk_size, total_docs - current_doc)
        docs = read_documents_batch(docs_path, current_doc, chunk_count)

        if not docs:
            break

        texts = [text for _, text in docs]
        embeddings = compute_embeddings_batch(model, texts, dim, batch_size)
        all_embeddings.append(embeddings)

        current_doc += len(docs)
        pbar.update(len(docs))

        # Save checkpoint
        if current_doc < total_docs and current_doc % chunk_size == 0:
            partial = np.vstack(all_embeddings)
            partial_path = os.path.join(data_path, "embeddings_partial.npy")
            np.save(partial_path, partial)
            save_checkpoint(data_path, current_doc, partial_path)
            elapsed = perf_counter() - t0
            docs_per_sec = current_doc / elapsed
            remaining = (total_docs - current_doc) / docs_per_sec
            print(f"\nCheckpoint at {current_doc:,} docs. ETA: {remaining/60:.1f} min")

    pbar.close()

    # Combine and save final embeddings
    final_embeddings = np.vstack(all_embeddings)
    print(f"\nFinal shape: {final_embeddings.shape}")

    # Write to binary format
    embedding_io = EmbeddingIO(cfg)
    embedding_io.write(final_embeddings, force=args.force or args.resume)

    # Clean up checkpoint
    checkpoint_path = get_checkpoint_path(data_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    partial_path = os.path.join(data_path, "embeddings_partial.npy")
    if os.path.exists(partial_path):
        os.remove(partial_path)

    elapsed = perf_counter() - t0
    mb = final_embeddings.nbytes / (1024 * 1024)
    print(f"Done! {final_embeddings.shape[0]:,} docs, {dim} dims, {mb:.1f} MB")
    print(f"Time: {elapsed/60:.1f} min ({final_embeddings.shape[0]/elapsed:.1f} docs/sec)")


if __name__ == "__main__":
    main()
