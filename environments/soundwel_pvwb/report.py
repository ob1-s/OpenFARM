#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

try:
    from sklearn.metrics import classification_report
except ImportError:
    raise ImportError("Please install scikit-learn: `uv pip install scikit-learn`")


def get_raw_prediction(row):
    """Safely extracts the raw VLM text, pre-parsed farm_pred, or catches API errors."""
    # 0. Catch actual infrastructure/API errors first!
    if row.get("error"):
        return "API_ERROR"

    # 1. If state_columns saved farm_pred successfully, use it!
    if "farm_pred" in row and row["farm_pred"] is not None:
        return str(row["farm_pred"])

    # 2. Otherwise, manually dig into the completion history
    completion = row.get("completion", [])
    if isinstance(completion, list):
        for msg in reversed(completion):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(p.get("text", "")) for p in content if isinstance(p, dict)
                    )
                return str(content)
    return ""


def infer_and_normalize(y_true_raw, y_pred_raw):
    """Auto-detects the task based on the ground truth and strictly normalizes predictions."""
    unique_true = set(y_true_raw)

    is_valence = "Positive" in unique_true or "Negative" in unique_true
    is_call_type = "HF" in unique_true or "LF" in unique_true

    y_pred_clean = []

    for p in y_pred_raw:
        p_up = str(p).upper()

        # Pass through API errors
        if p_up == "API_ERROR":
            y_pred_clean.append("API_ERROR")
            continue

        # Check if the model failed to output anything meaningful
        if not p_up.strip():
            y_pred_clean.append("FORMAT_FAILURE")
            continue

        if is_valence:
            if "POS" in p_up and "CONDITIONING" not in p_up:
                y_pred_clean.append("Positive")
            elif "NEG" in p_up and "CONDITIONING" not in p_up:
                y_pred_clean.append("Negative")
            else:
                y_pred_clean.append("FORMAT_FAILURE")

        elif is_call_type:
            if re.search(r"\bHF\b", p_up) or "HIGH FREQUENCY" in p_up:
                y_pred_clean.append("HF")
            elif re.search(r"\bLF\b", p_up) or "LOW FREQUENCY" in p_up:
                y_pred_clean.append("LF")
            else:
                y_pred_clean.append("FORMAT_FAILURE")

        else:
            # Fallback for Context or Age categories
            y_pred_clean.append(str(p).strip())

    return y_pred_clean


def main():
    parser = argparse.ArgumentParser(description="Generate Scientific F1-Score Report.")
    parser.add_argument(
        "results_path", type=str, help="Path to results.jsonl or run directory."
    )
    args = parser.parse_args()

    target_path = Path(args.results_path)
    if target_path.is_dir():
        target_path = target_path / "results.jsonl"

    if not target_path.exists():
        print(f"❌ Error: Could not find {target_path}")
        return

    y_true = []
    y_pred_raw = []

    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            y_true.append(str(row.get("answer", "")).strip())
            y_pred_raw.append(get_raw_prediction(row))

    # Auto-detect task and clean up predictions
    y_pred_clean = infer_and_normalize(y_true, y_pred_raw)

    print("=" * 60)
    print(" 📊 SCIENTIFIC EVALUATION REPORT")
    print("=" * 60)

    unique_true = sorted(list(set(y_true)))

    # Check for failures
    api_errors = sum(1 for p in y_pred_clean if p == "API_ERROR")
    format_failures = sum(1 for p in y_pred_clean if p == "FORMAT_FAILURE")

    if api_errors > 0:
        print(
            f"🚨  Critical: {api_errors} out of {len(y_pred_clean)} samples failed due to API/Infra errors."
        )
    if format_failures > 0:
        print(
            f"⚠️  Warning: {format_failures} out of {len(y_pred_clean)} samples failed parsing/formatting."
        )
    if api_errors > 0 or format_failures > 0:
        print("-" * 60)

    # Generate the report, restricted to valid Ground Truth classes
    print(
        classification_report(y_true, y_pred_clean, labels=unique_true, zero_division=0)
    )


if __name__ == "__main__":
    main()
