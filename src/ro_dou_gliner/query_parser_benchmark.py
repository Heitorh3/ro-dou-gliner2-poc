#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from .extractor import GlinerExtractor


@dataclass(frozen=True)
class TestCase:
    name: str
    query: str
    expected: dict[str, list[str]]


TEST_CASES = [
    TestCase(
        name="anm_alvara",
        query=(
            "publicações da Agência Nacional de Mineração " "sobre Alvará de Pesquisa"
        ),
        expected={
            "organization": [
                "Agência Nacional de Mineração",
            ],
            "official_publication": [
                "Alvará de Pesquisa",
            ],
        },
    ),
    TestCase(
        name="anm_codigo_mineracao",
        query=(
            "atos da Agência Nacional de Mineração "
            "relacionados ao Código de Mineração"
        ),
        expected={
            "organization": [
                "Agência Nacional de Mineração",
            ],
            "official_publication": [
                "Código de Mineração",
            ],
        },
    ),
    TestCase(
        name="janduci_alvara",
        query=("publicações de Janduci Dutra Fernandes " "sobre Alvará de Pesquisa"),
        expected={
            "person": [
                "Janduci Dutra Fernandes",
            ],
            "official_publication": [
                "Alvará de Pesquisa",
            ],
        },
    ),
    TestCase(
        name="janduci_codigo_mineracao",
        query=(
            "atos de Janduci Dutra Fernandes " "relacionados ao Código de Mineração"
        ),
        expected={
            "person": [
                "Janduci Dutra Fernandes",
            ],
            "official_publication": [
                "Código de Mineração",
            ],
        },
    ),
    TestCase(
        name="anm_coordenadora_tah",
        query=(
            "publicações da Agência Nacional de Mineração "
            "da Coordenadora de Autuação e Fiscalização da TAH"
        ),
        expected={
            "organization": [
                "Agência Nacional de Mineração",
            ],
            "position": [
                "Coordenadora de Autuação e Fiscalização da TAH",
            ],
        },
    ),
    TestCase(
        name="anm_coordenadora_tah_envolvendo",
        query=(
            "publicações da Agência Nacional de Mineração "
            "envolvendo a Coordenadora de Autuação e Fiscalização da TAH"
        ),
        expected={
            "organization": [
                "Agência Nacional de Mineração",
            ],
            "position": [
                "Coordenadora de Autuação e Fiscalização da TAH",
            ],
        },
    ),
    TestCase(
        name="anm_coordenadora_tah_na",
        query=(
            "publicações da Coordenadora de Autuação e Fiscalização da TAH "
            "na Agência Nacional de Mineração"
        ),
        expected={
            "organization": [
                "Agência Nacional de Mineração",
            ],
            "position": [
                "Coordenadora de Autuação e Fiscalização da TAH",
            ],
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark do parser de consultas GLiNER2."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/gliner.yaml"),
        help="Configuração GLiNER2.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Execuções medidas por consulta.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Execuções de warm-up antes da medição.",
    )

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def simplify_entities(
    result: dict[str, Any],
) -> dict[str, list[str]]:
    source = result.get("entities") or {}

    simplified: dict[str, list[str]] = {}

    for entity_type, values in source.items():
        extracted = []
        seen = set()

        for item in values:
            if not isinstance(item, dict):
                continue

            text = item.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip().rstrip(".,;:")

            if not text:
                continue

            key = text.casefold()

            if key in seen:
                continue

            seen.add(key)
            extracted.append(text)

        if extracted:
            simplified[entity_type] = extracted

    return simplified


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def evaluate(
    expected: dict[str, list[str]],
    actual: dict[str, list[str]],
) -> tuple[int, int, int]:
    expected_set = {
        (entity_type, normalize(value))
        for entity_type, values in expected.items()
        for value in values
    }

    actual_set = {
        (entity_type, normalize(value))
        for entity_type, values in actual.items()
        for value in values
    }

    true_positive = len(expected_set & actual_set)

    false_positive = len(actual_set - expected_set)

    false_negative = len(expected_set - actual_set)

    return (
        true_positive,
        false_positive,
        false_negative,
    )


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    print("Carregando GLiNER2...")

    load_start = time.perf_counter()

    extractor = GlinerExtractor.from_config(config)

    load_time = time.perf_counter() - load_start

    print(f"modelo_carregado={load_time:.3f}s")

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print(f"warmup={args.warmup}")

    warmup_query = TEST_CASES[0].query

    for _ in range(args.warmup):
        extractor.extract(warmup_query)

    print()
    print("=== Benchmark do parser ===")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_times: list[float] = []

    for number, test in enumerate(
        TEST_CASES,
        start=1,
    ):
        times = []
        result = {}

        for _ in range(args.runs):
            start = time.perf_counter()

            raw_result = extractor.extract(test.query)

            elapsed = time.perf_counter() - start

            times.append(elapsed)
            all_times.append(elapsed)

            result = simplify_entities(raw_result)

        tp, fp, fn = evaluate(
            expected=test.expected,
            actual=result,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = safe_div(
            tp,
            tp + fp,
        )

        recall = safe_div(
            tp,
            tp + fn,
        )

        f1 = safe_div(
            2 * precision * recall,
            precision + recall,
        )

        print()
        print(f"[{number}] {test.name}")
        print(f"consulta={test.query}")

        print(
            "esperado="
            + json.dumps(
                test.expected,
                ensure_ascii=False,
            )
        )

        print(
            "extraido="
            + json.dumps(
                result,
                ensure_ascii=False,
            )
        )

        print(f"TP={tp} FP={fp} FN={fn}")

        print(f"precision={precision:.3f} " f"recall={recall:.3f} " f"f1={f1:.3f}")

        print(f"latencia_media=" f"{mean(times) * 1000:.1f}ms")

        print(f"latencia_mediana=" f"{median(times) * 1000:.1f}ms")

        print(
            f"latencia_min="
            f"{min(times) * 1000:.1f}ms "
            f"latencia_max="
            f"{max(times) * 1000:.1f}ms"
        )

    # -----------------------------------------------------
    # Resultado global
    # -----------------------------------------------------

    precision = safe_div(
        total_tp,
        total_tp + total_fp,
    )

    recall = safe_div(
        total_tp,
        total_tp + total_fn,
    )

    f1 = safe_div(
        2 * precision * recall,
        precision + recall,
    )

    print()
    print("=== Resumo ===")

    print(f"consultas={len(TEST_CASES)}")

    print(f"execucoes_por_consulta={args.runs}")

    print(f"TP={total_tp} " f"FP={total_fp} " f"FN={total_fn}")

    print(f"precision={precision:.3f}")

    print(f"recall={recall:.3f}")

    print(f"f1={f1:.3f}")

    print(f"latencia_media=" f"{mean(all_times) * 1000:.1f}ms")

    print(f"latencia_mediana=" f"{median(all_times) * 1000:.1f}ms")

    print(f"latencia_min=" f"{min(all_times) * 1000:.1f}ms")

    print(f"latencia_max=" f"{max(all_times) * 1000:.1f}ms")


if __name__ == "__main__":
    main()
