import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))

log = logging.getLogger(__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Rows must be a non-empty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Row must be an object: {row!r}")
        customer_id = row["customer_id"]
        if not isinstance(customer_id, str) or not customer_id.strip():
            raise ValueError(f"Invalid customer_id: {customer_id!r}")
        if customer_id in seen:
            raise ValueError(f"Duplicate customer_id: {customer_id}")
        seen.add(customer_id)
        if row["business_date"] != day:
            raise ValueError(f"Wrong business_date for {customer_id}: {row['business_date']!r}")
        for name, (low, high) in zip(FEATURES, LIMITS):
            value = row[name]
            # type а не isinstance, потому что bool это подкласс int
            if type(value) not in (int, float):
                raise ValueError(f"Feature {name} of {customer_id} is not a number: {value!r}")
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"Feature {name} of {customer_id} is not finite: {value!r}")
            if not low <= value <= high:
                raise ValueError(f"Feature {name} of {customer_id} is out of range: {value!r}")
    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError(f"Invalid batch_size: {batch_size!r}")
    predictions = []
    for number, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start:start + batch_size]
        log.info("Batch %d: %d rows", number, len(batch))
        predictions.extend(predict(batch, model))
    return predictions


def validate_predictions(predictions, expected_ids):
    raise NotImplementedError("Задание 4: validate_predictions")


def replace_day(conn, day, model_version, predictions):
    raise NotImplementedError("Задание 5: replace_day")
