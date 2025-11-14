"""Prepare the Atticus contracts corpus for contract-specific midtraining.

This script streams the atticus_contracts split of pile-of-law/pile-of-law
from Hugging Face and repackages the data into ~250M-character parquet shards
so we can reuse the same streaming infrastructure as base pretraining.
"""

import argparse
import os
import time
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat.common import get_base_dir, print0
from nanochat.dataset import CONTRACT_DATA_DIR

DEFAULT_CHARS_PER_SHARD = 250_000_000
DEFAULT_TARGET_SHARDS = 110
ROW_GROUP_SIZE = 1024
SHARD_TEMPLATE = "shard_{:05d}.parquet"


def iter_contract_documents() -> Iterable[str]:
    """Yield raw text fields from the atticus_contracts subset."""

    dataset = load_dataset(
        "pile-of-law/pile-of-law",
        name="atticus_contracts",
        split="train",
        streaming=True,
    )
    for example in dataset:
        text = example.get("text") or ""
        if not text:
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        yield cleaned


def discover_next_shard_index(max_shards: int) -> int:
    """Return the first shard index we still need to create."""

    existing = [
        fname for fname in os.listdir(CONTRACT_DATA_DIR)
        if fname.endswith(".parquet") and fname.startswith("shard_")
    ]
    if not existing:
        return 0
    indices = []
    for fname in existing:
        stem = fname.replace("shard_", "").replace(".parquet", "")
        try:
            indices.append(int(stem))
        except ValueError:
            continue
    if not indices:
        return 0
    highest = max(indices)
    next_idx = highest + 1
    if next_idx >= max_shards:
        return max_shards
    return next_idx


def write_shard(texts: List[str], shard_index: int) -> str:
    """Write the buffered texts to disk as a single parquet shard."""

    output_path = os.path.join(CONTRACT_DATA_DIR, SHARD_TEMPLATE.format(shard_index))
    tmp_path = output_path + ".tmp"
    table = pa.Table.from_pydict({"text": texts})
    pq.write_table(
        table,
        tmp_path,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    os.replace(tmp_path, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare contract parquet shards")
    parser.add_argument(
        "--chars-per-shard",
        type=int,
        default=DEFAULT_CHARS_PER_SHARD,
        help="Approximate number of characters per output shard",
    )
    parser.add_argument(
        "--target-shards",
        type=int,
        default=DEFAULT_TARGET_SHARDS,
        help="Stop once this many shards exist",
    )
    args = parser.parse_args()

    os.makedirs(CONTRACT_DATA_DIR, exist_ok=True)
    print0(f"Contract corpus destination: {CONTRACT_DATA_DIR}")
    print0(f"Chars per shard target: {args.chars_per_shard:,}")
    print0(f"Target shard count: {args.target_shards}")

    # if we already produced enough shards, bail early
    existing_shards = [
        fname for fname in os.listdir(CONTRACT_DATA_DIR)
        if fname.endswith(".parquet") and fname.startswith("shard_")
    ]
    if len(existing_shards) >= args.target_shards:
        print0(
            f"Found {len(existing_shards)} shards >= target {args.target_shards}. Nothing to do."
        )
        return

    shard_index = discover_next_shard_index(args.target_shards)
    if shard_index >= args.target_shards:
        print0(
            f"Shard numbering already reaches target ({shard_index} >= {args.target_shards})."
        )
        return

    documents = iter_contract_documents()
    buffer: List[str] = []
    shard_characters = 0
    total_chars = 0
    total_docs = 0
    shard_timer = time.time()

    for doc in documents:
        buffer.append(doc)
        doc_len = len(doc)
        shard_characters += doc_len
        total_docs += 1
        if shard_characters < args.chars_per_shard:
            continue
        if len(buffer) % ROW_GROUP_SIZE != 0:
            continue

        shard_path = os.path.join(CONTRACT_DATA_DIR, SHARD_TEMPLATE.format(shard_index))
        if os.path.exists(shard_path):
            print0(f"Skipping existing {shard_path}")
        else:
            write_shard(buffer, shard_index)
            elapsed = time.time() - shard_timer
            total_chars += shard_characters
            remaining = args.target_shards - (shard_index + 1)
            print0(
                f"Wrote {shard_path} | docs: {len(buffer)} | chars: {shard_characters:,} | "
                f"total chars: {total_chars:,} | shards left: {max(0, remaining)} | dt: {elapsed:.2f}s"
            )
        shard_index += 1
        buffer = []
        shard_characters = 0
        shard_timer = time.time()
        if shard_index >= args.target_shards:
            break

    # Flush any remaining buffer if we still owe a shard and have data
    if buffer and shard_index < args.target_shards:
        shard_path = os.path.join(CONTRACT_DATA_DIR, SHARD_TEMPLATE.format(shard_index))
        if os.path.exists(shard_path):
            print0(f"Skipping existing {shard_path}")
        else:
            write_shard(buffer, shard_index)
            total_chars += shard_characters
            print0(
                f"Wrote final partial shard {shard_path} | docs: {len(buffer)} | chars: {shard_characters:,}"
            )
        shard_index += 1

    print0(
        f"Finished contract corpus preparation. Produced shards up to index {shard_index - 1}."
    )


if __name__ == "__main__":
    main()
