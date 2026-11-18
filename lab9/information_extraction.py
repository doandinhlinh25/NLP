"""Extract structured named entities from user-provided text with the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import pipeline

MODEL_DIR = Path(__file__).resolve().parent / "checkpoints" / "token_classifier"
ENTITY_MAP = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISCELLANEOUS"}


def extract_entities(text: str, model_path: Path = MODEL_DIR) -> dict[str, list[str]]:
    """Return deduplicated entity strings grouped by readable entity category."""
    ner = pipeline("token-classification", model=str(model_path), tokenizer=str(model_path), aggregation_strategy="simple")
    extracted: dict[str, list[str]] = {category: [] for category in ENTITY_MAP.values()}
    for item in ner(text):
        label = str(item["entity_group"])
        category = ENTITY_MAP.get(label)
        entity = str(item["word"]).strip()
        if category and entity and entity not in extracted[category]:
            extracted[category].append(entity)
    return {category: entities for category, entities in extracted.items() if entities}


def main() -> None:
    """Read text from the command line and print its structured entities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to process with the trained NER model.")
    arguments = parser.parse_args()
    print(extract_entities(arguments.text))


if __name__ == "__main__":
    main()
