#!/usr/bin/env python3

from __future__ import annotations

import json
import time

from src.ro_dou_gliner.extractor import GlinerExtractor

TEST_QUERIES = [
    "publicações do Banco Interamericano de Desenvolvimento relacionadas a Mato Grosso do Sul",
    "atos envolvendo o Ministério da Fazenda em Brasília",
    "publicações sobre João da Silva no Ministério da Gestão e da Inovação em Serviços Públicos",
    "atos relacionados ao processo 17944.006645/2025-65",
    "publicações referentes ao SIAPE 2948141",
    "nomeações para o cargo de Coordenador de Compras e Contratos",
    "atos relacionados ao cargo CCE 1.10",
    "publicações da Universidade Federal de Uberlândia em Uberlândia",
    "portarias envolvendo o Ministério da Educação",
    "atos envolvendo Maria da Silva em Minas Gerais",
]


CONFIG = {
    "gliner": {
        "enabled": True,
        "model": "fastino/gliner2.5-multi-v1",
        "threshold": 0.45,
        "include_confidence": True,
        "include_spans": True,
        "chunking": {"enabled": False},
        "entities": {
            "person": ("Nome completo de pessoa física mencionada no ato."),
            "organization": (
                "Nome de órgão público, entidade, instituição, "
                "universidade, empresa ou unidade administrativa."
            ),
            "position": (
                "Nome de cargo, emprego público, função ou " "função comissionada."
            ),
            "position_code": (
                "Código ou símbolo de cargo ou função, como "
                "CCE 1.10, FCE 2.10, DAS 101.4, CD-02 ou FG-1."
            ),
            "location": (
                "Nome próprio de localização geográfica real, "
                "como município, estado ou país."
            ),
            "process": (
                "Número ou identificação de processo administrativo " "ou judicial."
            ),
            "registration": (
                "Número de matrícula ou identificação funcional "
                "de servidor público, incluindo SIAPE."
            ),
            "legal_act": (
                "Identificação de ato normativo ou administrativo, "
                "como Portaria, Decreto, Resolução ou Edital."
            ),
        },
    }
}


def simplify(result: dict) -> dict[str, list[str]]:
    entities = result.get("entities") or {}

    simplified = {}

    for entity_type, values in entities.items():
        texts = []

        for value in values:
            if isinstance(value, dict):
                text = value.get("text")

                if text:
                    texts.append(text)

        if texts:
            simplified[entity_type] = texts

    return simplified


def main() -> None:
    extractor = GlinerExtractor.from_config(config=CONFIG)

    print("=== Benchmark de consultas GLiNER2 ===")
    print()

    total_time = 0.0

    for number, query in enumerate(TEST_QUERIES, start=1):
        start = time.perf_counter()

        result = extractor.extract(query)

        elapsed = time.perf_counter() - start
        total_time += elapsed

        entities = simplify(result)

        print(f"[{number}] {query}")
        print(f"tempo={elapsed:.3f}s")

        print(
            json.dumps(
                entities,
                ensure_ascii=False,
                indent=2,
            )
        )

        print("-" * 70)

    print()
    print("=== Resumo ===")
    print(f"consultas={len(TEST_QUERIES)}")
    print(f"tempo_total={total_time:.3f}s")
    print(f"tempo_medio={total_time / len(TEST_QUERIES):.3f}s")


if __name__ == "__main__":
    main()
