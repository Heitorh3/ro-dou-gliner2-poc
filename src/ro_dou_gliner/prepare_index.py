#!/usr/bin/env python3

"""
Prepara os documentos enriquecidos pelo GLiNER2 para indexação no OpenSearch.

Entrada:
    JSONL gerado pelo benchmark GLiNER2.

Saída:
    JSONL contendo os campos originais relevantes e uma estrutura simplificada
    de entidades para pesquisa no OpenSearch.

Exemplo:

    uv run python prepare_index.py \
        output/gliner-results-500.jsonl \
        --output output/opensearch-documents.jsonl

Estrutura gerada:

{
    "id": 123,
    "pubdate": "2026-09-04",
    "pubname": "DO1",
    "arttype": "Portaria",
    "artcategory": "...",
    "identifica": "...",
    "assina": "...",
    "texto_plain": "...",
    "entities": {
        "person": [...],
        "organization": [...],
        "position": [...],
        "position_code": [...],
        "location": [...],
        "process": [...],
        "registration": [...],
        "legal_act": [...]
    }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ENTITY_TYPES = (
    "person",
    "organization",
    "position",
    "position_code",
    "location",
    "process",
    "registration",
    "legal_act",
)


DOCUMENT_FIELDS = (
    "id",
    "_opensearch_id",
    "pubdate",
    "pubname",
    "arttype",
    "artcategory",
    "identifica",
    "assina",
    "texto_plain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara documentos enriquecidos pelo GLiNER2 "
            "para indexação no OpenSearch."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Arquivo JSONL gerado pelo benchmark GLiNER2.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/opensearch-documents.jsonl"),
        help=("Arquivo JSONL de saída. " "Padrão: output/opensearch-documents.jsonl"),
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help=(
            "Confiança mínima para manter uma entidade. " "Padrão: 0.0 (não filtra)."
        ),
    )

    return parser.parse_args()


def normalize_text(value: str) -> str:
    """
    Normalização mínima para deduplicação.

    Mantemos o texto original no documento final, usando a versão
    normalizada apenas como chave de comparação.
    """
    return " ".join(value.split()).casefold()


def extract_entity_values(
    document: dict[str, Any],
    min_confidence: float = 0.0,
) -> dict[str, list[str]]:
    """
    Converte a saída detalhada do GLiNER2 em listas simples.

    Entrada:

        "person": [
            {
                "text": "JOÃO DA SILVA",
                "confidence": 0.99,
                "start": 10,
                "end": 23
            }
        ]

    Saída:

        "person": [
            "JOÃO DA SILVA"
        ]
    """

    gliner = document.get("gliner") or {}
    result = gliner.get("result") or {}
    source_entities = result.get("entities") or {}

    entities: dict[str, list[str]] = {}

    for entity_type in ENTITY_TYPES:
        values = source_entities.get(entity_type) or []

        extracted: list[str] = []
        seen: set[str] = set()

        for entity in values:
            if not isinstance(entity, dict):
                continue

            text = entity.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            confidence = entity.get("confidence")

            if confidence is not None and float(confidence) < min_confidence:
                continue

            normalized = normalize_text(text)

            if normalized in seen:
                continue

            seen.add(normalized)
            extracted.append(text)

        entities[entity_type] = extracted

    return entities


def prepare_document(
    document: dict[str, Any],
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """
    Cria o documento que será posteriormente indexado no OpenSearch.
    """

    prepared = {
        field: document.get(field) for field in DOCUMENT_FIELDS if field in document
    }

    prepared["entities"] = extract_entity_values(
        document,
        min_confidence=min_confidence,
    )

    return prepared


def process_file(
    input_path: Path,
    output_path: Path,
    min_confidence: float,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    errors = 0
    entity_counts = {entity_type: 0 for entity_type in ENTITY_TYPES}

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as input_file,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file,
    ):
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

                prepared = prepare_document(
                    document,
                    min_confidence=min_confidence,
                )

                output_file.write(
                    json.dumps(
                        prepared,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                processed += 1

                for entity_type in ENTITY_TYPES:
                    entity_counts[entity_type] += len(prepared["entities"][entity_type])

            except Exception as exc:
                errors += 1

                print(f"[ERRO] linha={line_number}: {exc}")

    print()
    print("=== Preparação para OpenSearch ===")
    print(f"documentos_processados={processed}")
    print(f"erros={errors}")
    print(f"min_confidence={min_confidence}")
    print(f"saida={output_path}")

    print()
    print("=== Entidades ===")

    for entity_type, count in entity_counts.items():
        print(f"{entity_type:15} {count:6}")


def main() -> None:
    args = parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
