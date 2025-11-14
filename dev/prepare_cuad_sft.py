"""Convert the CUAD QA dataset into nanochat-compatible chat conversations."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional

from datasets import load_dataset

from nanochat.common import get_base_dir, print0

OUTPUT_FILENAME = "cuad_sft_conversations.jsonl"


QUESTION_KEYS = ["question", "question_text", "prompt"]
ANSWER_KEYS = [
    "answer",
    "answer_text",
    "label",
    "short_answer",
]
CONTEXT_KEYS = ["context", "passage", "clause", "text"]


def extract_first_nonempty(options: Iterable[Optional[str]]) -> Optional[str]:
    for value in options:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return None


def build_conversation(example: Dict) -> Optional[List[Dict[str, str]]]:
    question = extract_first_nonempty(example.get(key) for key in QUESTION_KEYS)
    if question is None:
        return None
    context_candidate = extract_first_nonempty(example.get(key) for key in CONTEXT_KEYS)

    answer = None
    answers_field = example.get("answers")
    if isinstance(answers_field, dict):
        answer = extract_first_nonempty(answers_field.get("text", []))
    if answer is None:
        answer = extract_first_nonempty(example.get(key) for key in ANSWER_KEYS)
    if answer is None:
        return None

    user_preamble = (
        "You are an expert contract analyst specializing in procurement and supply chain risk. "
        "Review the clause and respond with a concise, factual answer."
    )
    if context_candidate:
        clause_section = f"Contract clause:\n{context_candidate}\n\n"
    else:
        clause_section = ""
    user_content = f"{user_preamble}\n\n{clause_section}Question:\n{question}"
    assistant_content = answer
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def main():
    parser = argparse.ArgumentParser(description="Prepare CUAD SFT conversations")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validation"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (defaults to ~/.cache/nanochat/cuad_sft_conversations.jsonl)",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    output_path = args.output or os.path.join(base_dir, OUTPUT_FILENAME)
    tmp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_examples = 0
    written_examples = 0
    skipped_examples = 0

    with open(tmp_path, "w", encoding="utf-8") as fout:
        for split in args.splits:
            print0(f"Loading CUAD split: {split}")
            dataset = load_dataset("theatticusproject/cuad-qa", split=split)
            for example in dataset:
                total_examples += 1
                conversation = build_conversation(example)
                if conversation is None:
                    skipped_examples += 1
                    continue
                fout.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                written_examples += 1

    os.replace(tmp_path, output_path)
    print0(
        f"Saved {written_examples} CUAD conversations (skipped {skipped_examples} / {total_examples}) to {output_path}"
    )


if __name__ == "__main__":
    main()
