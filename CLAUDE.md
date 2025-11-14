# CLAUDE Project Brief

## Overview
This fork, `nanochat-contracts`, repurposes the original nanochat pipeline for procurement and contract analytics. The tokenizer, base pretraining, evaluation harness, and serving stack remain identical to upstream so we inherit the same “clean, minimal, hackable” footprint. The differentiator lies in the midtraining and supervised fine-tuning phases, which now focus on real-world legal clauses and Q&A pairs.

## Domain datasets
- **pile-of-law/pile-of-law (atticus_contracts)** – streamed via `dev/prepare_contract_corpus.py`, repackaged into ~250M character shards with zstd compression, and consumed by `scripts/contract_mid_train.py` for midtraining.
- **theatticusproject/cuad-qa** – converted to chat-format JSONL with `dev/prepare_cuad_sft.py` and loaded by `scripts/chat_sft.py` for contract-focused SFT.

Both corpora land under `~/.cache/nanochat/` so tooling can reuse the standard cache directory.

## Build / training checklist
1. **Environment:** run `uv sync --extra gpu`, install Rust dependencies, and activate `.venv` (unchanged from upstream).
2. **Base pretraining:** execute `python -m nanochat.dataset -n 16`, start the longer download with `-n 800`, then run `torchrun ... -m scripts.base_train` followed by `scripts.base_loss` and `scripts.base_eval`.
3. **Contract corpus prep:** `python dev/prepare_contract_corpus.py` writes parquet shards to `~/.cache/nanochat/contract_data` with zstd level 3 compression so the tokenizer/dataloader can stream them efficiently.
4. **Contract midtraining:** launch `torchrun --standalone --nproc_per_node=8 -m scripts.contract_mid_train -- --device_batch_size=8`. The script reuses the Muon + AdamW optimizers, keeps the same LR schedule, and automatically derives the number of iterations from the available shards unless overridden.
5. **CUAD SFT:** `python dev/prepare_cuad_sft.py` -> `torchrun ... -m scripts.chat_sft`. The `CustomJSON` task now points to `~/.cache/nanochat/cuad_sft_conversations.jsonl`.
6. **Evaluations & report:** run `scripts.chat_eval` after mid and SFT, then `python -m nanochat.report generate` and optionally `python -m scripts.chat_web` for the UI.

The full $1000 pipeline is scripted in `run1000.sh` with the new prep/training commands included.

## Hybrid training approach
We keep the strong generalist foundation from FineWeb-Edu so the model avoids catastrophic forgetting and still performs on open-domain reasoning benches. Midtraining saturates the network with real contracts, ensuring embeddings pay special attention to boilerplate, clauses, and procurement jargon. Finally, CUAD-style supervision sharpens the chat interface so outputs are framed as compliance answers. Combined, this hybrid strategy yields a compact assistant that is both grounded in general language modeling and calibrated for contract review workflows.
