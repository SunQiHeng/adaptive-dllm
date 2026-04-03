#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def resolve_data_files(data_path: str) -> list[str]:
    p = Path(data_path)
    if p.is_file():
        return [str(p.resolve())]
    if not p.is_dir():
        raise FileNotFoundError(f"GPQA data path not found: {data_path}")

    files = sorted(
        str(x.resolve())
        for x in p.rglob("*")
        if x.is_file() and x.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found under GPQA data path: {data_path}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local lm-eval GPQA task config.")
    parser.add_argument("--data_path", required=True, help="Local .json/.jsonl file or directory for GPQA rows.")
    parser.add_argument("--output_dir", required=True, help="Directory to write the generated task config into.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    src_utils = Path(__file__).resolve().parent / "gpqa_local" / "utils.py"
    dst_utils = out_dir / "utils.py"
    shutil.copyfile(src_utils, dst_utils)

    data_files = resolve_data_files(args.data_path)
    shared_yaml = out_dir / "_gpqa_local_n_shot_yaml"
    task_yaml = out_dir / "gpqa_main_n_shot_local.yaml"

    shared_yaml.write_text(
        "\n".join(
            [
                "dataset_path: json",
                "dataset_name: null",
                "dataset_kwargs:",
                "  data_files:",
                f"    train: {json.dumps(data_files)}",
                "tag: gpqa",
                "output_type: multiple_choice",
                "process_docs: !function utils.process_docs",
                "training_split: train",
                "validation_split: train",
                "test_split: null",
                'description: "Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.\\n"',
                'doc_to_text: "Question: {{Question if Question is defined else question}}\\nChoices:\\n(A) {{choice1}}\\n(B) {{choice2}}\\n(C) {{choice3}}\\n(D) {{choice4}}\\nAnswer:"',
                "doc_to_target: answer",
                'doc_to_choice: ["(A)", "(B)", "(C)", "(D)"]',
                "metric_list:",
                "  - metric: acc",
                "    aggregation: mean",
                "    higher_is_better: true",
                "  - metric: acc_norm",
                "    aggregation: mean",
                "    higher_is_better: true",
                "metadata:",
                "  version: 2.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    task_yaml.write_text(
        "\n".join(
            [
                "dataset_name: null",
                "include: _gpqa_local_n_shot_yaml",
                "task: gpqa_main_n_shot_local",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(str(out_dir))


if __name__ == "__main__":
    main()
