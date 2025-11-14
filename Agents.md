# Agents Playbook

## Mission
Operate `nanochat-contracts` as a specialized LLM pipeline for contract, procurement, and supply-chain intelligence. Every change should preserve the upstream nanochat minimalism while keeping this fork aligned with SamerGMTM22/My-nanochat.

## Source control guidelines
- Treat `main` as the integration branch; create topic branches per feature (`feat/contract-mid`, `fix/cuad-format`, etc.).
- Rebase onto upstream `karpathy/nanochat` periodically but only after verifying compatibility with the contract-specific mods.
- Never rewrite history on shared branches. Use merge requests or PRs against the fork (GitHub repo `samerGMTM22/My-nanochat`).
- Commit messages: short imperative subject, optional body describing datasets/scripts touched (e.g., `Add CUAD prep + chat_sft hook`).

## Workflow best practices
1. **Environment parity:** Always run `uv sync --extra gpu` and `uv run maturin develop --manifest-path rustbpe/Cargo.toml` before training.
2. **Data caching:** Keep `~/.cache/nanochat/contract_data` and `~/.cache/nanochat/cuad_sft_conversations.jsonl` up-to-date via `dev/prepare_contract_corpus.py` and `dev/prepare_cuad_sft.py`. Avoid checking datasets into git.
3. **Training:** Use `scripts/contract_mid_train.py` for midtraining and `scripts/chat_sft.py` for SFT. Match `device_batch_size` with base training when launching via `torchrun`.
4. **Evaluation:** Run `scripts.base_eval`, `scripts.chat_eval`, and `python -m nanochat.report generate` after each stage. Record WANDB runs by setting `WANDB_RUN`.
5. **Testing:** For code changes, run targeted unit tests under `tests/` plus quick smoke tests of new scripts (`python dev/prepare_contract_corpus.py --target-shards 1`).
6. **Documentation:** Update `README.md`, `CLAUDE.md`, and this file whenever workflows change.

## Coding standards
- Match upstream formatting (PEP8, docstrings, inline comments only where non-obvious logic exists).
- Prefer reusing nanochat helpers (`get_base_dir`, `tokenizing_distributed_data_loader`) over duplicating logic.
- Maintain DDP compatibility; gate logging with `print0`.
- Keep configs override-able via `nanochat/configurator.py` (all tunables should be globals at top of scripts).

## Deployment checklist
- `python dev/prepare_contract_corpus.py`
- `torchrun ... scripts.contract_mid_train`
- `python dev/prepare_cuad_sft.py`
- `torchrun ... scripts.chat_sft`
- `python -m nanochat.report generate`
- `python -m scripts.chat_web`

Stick to this playbook so collaborating agents can jump in, understand the contract-specific extensions, and safely evolve the fork.
