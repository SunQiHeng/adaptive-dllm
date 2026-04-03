from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import get_dataset_config_names, load_dataset


_DATASET_ALIASES = {
    "gsm8k": "gsm8k",
    "minerva_math": "minerva_math",
    "math": "minerva_math",
    "mmlu": "mmlu",
    "cmmlu": "cmmlu",
    "ceval-valid": "ceval-valid",
    "ceval": "ceval-valid",
    "gpqa_main_n_shot": "gpqa_main_n_shot",
    "gpqa_main": "gpqa_main_n_shot",
    "gpqa": "gpqa_main_n_shot",
    "humaneval": "humaneval",
    "mbpp": "mbpp",
    "nemotron": "nemotron",
}


def _resolve_local_data_paths(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    paths: List[str] = []
    for root, _, files in os.walk(data_path):
        for fn in sorted(files):
            if fn.endswith(".jsonl") or fn.endswith(".json"):
                paths.append(os.path.join(root, fn))
    if not paths:
        raise FileNotFoundError(f"No .json or .jsonl files found under directory: {data_path}")
    return sorted(paths)


def _normalize_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            out.append(dict(item["row"]))
        elif isinstance(item, dict):
            out.append(dict(item))
        else:
            raise ValueError(f"Expected dict-like row, got {type(item)}")
    return out


def normalize_dataset_name(dataset_name: str) -> str:
    key = str(dataset_name).strip().lower()
    if key not in _DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset/task alias: {dataset_name!r}")
    return _DATASET_ALIASES[key]


def _normalize_multichoice_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(row.get("choices"), list) and row.get("answer", None) is not None:
        out = dict(row)
        out["question"] = str(row.get("question", row.get("prompt", "")))
        return out

    question = (
        row.get("question")
        or row.get("Question")
        or row.get("prompt")
        or row.get("Prompt")
        or row.get("query")
        or row.get("Query")
    )
    if question is None:
        raise ValueError("MCQ row missing question/prompt field.")

    if all(k in row for k in ("A", "B", "C", "D")):
        choices = [row["A"], row["B"], row["C"], row["D"]]
        answer = row.get("answer", row.get("Answer", row.get("target", row.get("label"))))
        return {"question": str(question), "choices": [str(x) for x in choices], "answer": answer}

    correct = row.get("Correct Answer", row.get("correct_answer", row.get("correct")))
    incorrect = [
        row.get("Incorrect Answer 1", row.get("incorrect_answer_1", row.get("incorrect1"))),
        row.get("Incorrect Answer 2", row.get("incorrect_answer_2", row.get("incorrect2"))),
        row.get("Incorrect Answer 3", row.get("incorrect_answer_3", row.get("incorrect3"))),
    ]
    if correct is not None and all(x is not None for x in incorrect):
        options = [str(correct), str(incorrect[0]), str(incorrect[1]), str(incorrect[2])]
        seed_str = str(question)[:200] + str(correct)[:50]
        rng = random.Random(seed_str)
        indices = list(range(4))
        rng.shuffle(indices)
        choices = [options[i] for i in indices]
        answer = indices.index(0)
        return {"question": str(question), "choices": choices, "answer": answer}

    raise ValueError("MCQ row missing recognizable choice fields.")


def _normalize_math_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if row.get("question", None) is not None and row.get("answer", None) is not None:
        return {"question": str(row["question"]), "answer": str(row["answer"])}

    question = row.get("problem", row.get("Problem", row.get("content", row.get("prompt", row.get("question")))))
    answer = row.get("solution", row.get("Solution", row.get("target", row.get("answer"))))
    if question is None or answer is None:
        raise ValueError("Math row missing recognizable question/answer fields.")
    return {"question": str(question), "answer": str(answer)}


def _normalize_code_row(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "humaneval":
        prompt = row.get("prompt", row.get("text", row.get("question", "")))
        solution = row.get("canonical_solution", row.get("solution", row.get("code", "")))
        return {"prompt": str(prompt), "canonical_solution": str(solution)}

    prompt = row.get("prompt", row.get("text", row.get("question", "")))
    solution = row.get("code", row.get("canonical_solution", row.get("solution", "")))
    if prompt is None or solution is None:
        raise ValueError("MBPP row missing recognizable prompt/code fields.")
    return {"prompt": str(prompt), "canonical_solution": str(solution)}


def canonicalize_row_for_dataset(row: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "nemotron":
        return dict(row)
    if dataset_name in {"mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot"}:
        return _normalize_multichoice_row(row)
    if dataset_name in {"gsm8k", "minerva_math"}:
        return _normalize_math_row(row)
    if dataset_name in {"humaneval", "mbpp"}:
        return _normalize_code_row(row, dataset_name)
    raise ValueError(f"Unsupported dataset for canonicalization: {dataset_name}")


def _load_local_json(path: str, split: Optional[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return _normalize_rows(obj)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported JSON root type in {path}: {type(obj)}")

    if split and isinstance(obj.get(split), list):
        return _normalize_rows(obj[split])
    if isinstance(obj.get("rows"), list):
        return _normalize_rows(obj["rows"])
    if isinstance(obj.get("data"), list):
        return _normalize_rows(obj["data"])
    if isinstance(obj.get("examples"), list):
        return _normalize_rows(obj["examples"])

    return _normalize_rows([obj])


def _load_local_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                rows.append(dict(item["row"]))
            elif isinstance(item, dict):
                rows.append(dict(item))
            else:
                raise ValueError(f"Expected dict-like row in {path}, got {type(item)}")
    return rows


def load_local_rows(
    data_path: str,
    *,
    max_samples: int,
    data_seed: int,
    split: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _resolve_local_data_paths(data_path):
        if path.endswith(".jsonl"):
            rows.extend(_load_local_jsonl(path))
        elif path.endswith(".json"):
            rows.extend(_load_local_json(path, split))
        else:
            raise ValueError(f"Unsupported file extension for local data: {path}")

    if len(rows) > 1:
        rng = random.Random(int(data_seed))
        rng.shuffle(rows)

    if int(max_samples) > 0:
        rows = rows[: min(int(max_samples), len(rows))]
    if dataset_name:
        rows = [canonicalize_row_for_dataset(row, dataset_name) for row in rows]
    return rows


def load_hf_rows(
    dataset_name: str,
    *,
    dataset_config: str,
    split: str,
    max_samples: int,
    data_seed: int,
    samples_per_category: int = 0,
    nemotron_pool_per_category: int = 0,
    nemotron_categories: str = "",
) -> List[Dict[str, Any]]:
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", dataset_config or "main", split=split)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name == "minerva_math":
        ds = load_dataset("knoveleng/Minerva-Math", split=split)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name == "mmlu":
        subject = dataset_config if dataset_config not in {"", "main"} else "all"
        ds = load_dataset("cais/mmlu", subject, split=split)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name == "gpqa_main_n_shot":
        subject = dataset_config if dataset_config not in {"", "main"} else "gpqa_main"
        ds = load_dataset("Idavidrein/gpqa", subject, split=split)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name == "mbpp":
        config = dataset_config if dataset_config not in {"", "main"} else "sanitized"
        ds = load_dataset("google-research-datasets/mbpp", config, split=split)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    if dataset_name in {"cmmlu", "ceval-valid"}:
        return _load_multiconfig_mcq_rows(
            dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_samples=max_samples,
            data_seed=data_seed,
        )

    if dataset_name == "nemotron":
        cats = [c.strip() for c in str(nemotron_categories).split(",") if c.strip()]
        rows: List[Dict[str, Any]] = []
        pool_per_category = max(int(samples_per_category), int(nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf: List[Dict[str, Any]] = []
            for sample in stream:
                buf.append(dict(sample) if isinstance(sample, dict) else sample)
                if len(buf) >= int(pool_per_category):
                    break
            if len(buf) > 1:
                rng = random.Random(_stable_int_seed(int(data_seed), int(cat_idx)))
                rng.shuffle(buf)
            rows.extend(buf[: min(int(samples_per_category), len(buf))])
        if len(rows) > 1:
            rng = random.Random(int(data_seed))
            rng.shuffle(rows)
        rows = rows[: min(int(max_samples), len(rows))]
        return rows

    raise ValueError(f"Unsupported dataset_name for HF loading: {dataset_name}")


def _sample_indices(total: int, max_samples: int, data_seed: int) -> List[int]:
    indices = list(range(total))
    if len(indices) > 1:
        rng = random.Random(int(data_seed))
        rng.shuffle(indices)
    if int(max_samples) > 0:
        indices = indices[: min(int(max_samples), len(indices))]
    return indices


def _stable_int_seed(*parts: int) -> int:
    x = 0x9E3779B97F4A7C15
    for p in parts:
        v = int(p) & ((1 << 64) - 1)
        x ^= v + 0x9E3779B97F4A7C15 + ((x << 6) & ((1 << 64) - 1)) + (x >> 2)
        x &= (1 << 64) - 1
    return int(x % (2**31 - 1))


def _load_multiconfig_mcq_rows(
    dataset_name: str,
    *,
    dataset_config: str,
    split: str,
    max_samples: int,
    data_seed: int,
) -> List[Dict[str, Any]]:
    if dataset_name == "cmmlu":
        hf_id = "haonan-li/cmmlu"
    elif dataset_name == "ceval-valid":
        hf_id = "ceval/ceval-exam"
    else:
        raise ValueError(f"Unsupported multi-config dataset: {dataset_name}")

    split_name = split
    if dataset_name == "ceval-valid" and split in {"test", "main", ""}:
        split_name = "val"

    config = str(dataset_config).strip()
    if config not in {"", "main", "all"}:
        ds = load_dataset(hf_id, config, split=split_name)
        rows = [dict(ds[i]) for i in _sample_indices(len(ds), max_samples, data_seed)]
        return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]

    config_names = [c for c in get_dataset_config_names(hf_id) if c and c != "default"]
    if not config_names:
        raise RuntimeError(f"No dataset configs found for {hf_id}")

    rows: List[Dict[str, Any]] = []
    per_cfg = max(1, int(max_samples) // max(1, len(config_names)))
    for cfg_idx, cfg in enumerate(config_names):
        ds = load_dataset(hf_id, cfg, split=split_name)
        idx = _sample_indices(len(ds), per_cfg, _stable_int_seed(int(data_seed), cfg_idx))
        for i in idx:
            row = dict(ds[int(i)])
            row["subject"] = row.get("subject", cfg)
            rows.append(row)
    if len(rows) > 1:
        rng = random.Random(int(data_seed))
        rng.shuffle(rows)
    rows = rows[: min(int(max_samples), len(rows))]
    return [canonicalize_row_for_dataset(row, dataset_name) for row in rows]
