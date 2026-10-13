from __future__ import annotations

import argparse
import json
import time
from typing import Any, Mapping

from opensearchpy import OpenSearch

from .extractor import GlinerExtractor

DEFAULT_INDEX = "rodou-gliner"

ENTITY_BOOSTS = {
    "person": 5.0,
    "organization": 5.0,
    "position": 3.0,
    "position_code": 4.0,
    "location": 3.0,
    "process": 5.0,
    "registration": 5.0,
    "legal_act": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca documentos usando entidades extraídas pelo GLiNER2."
    )
    parser.add_argument("query", help="Consulta em linguagem natural.")
    parser.add_argument(
        "--host", default="http://localhost:9200", help="Endereço do OpenSearch."
    )
    parser.add_argument(
        "--index", default=DEFAULT_INDEX, help=f"Índice. Padrão: {DEFAULT_INDEX}"
    )
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument(
        "--size", type=int, default=10, help="Número máximo de resultados."
    )
    parser.add_argument(
        "--show-dsl", action="store_true", help="Exibe a query OpenSearch gerada."
    )
    parser.add_argument(
        "--config",
        default="config/gliner.yaml",
        help="Arquivo YAML de configuração do GLiNER2.",
    )
    return parser.parse_args()


def load_config(path: str) -> Mapping[str, Any]:
    """Carrega o mesmo arquivo YAML de configuração usado pelo projeto."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML não está instalado. Execute: uv add pyyaml") from exc

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuração inválida em {path}: esperado um objeto YAML.")

    return config


def create_client(host: str, username: str | None, password: str | None) -> OpenSearch:
    kwargs: dict[str, Any] = {"hosts": [host]}
    if username and password:
        kwargs["http_auth"] = (username, password)
    return OpenSearch(**kwargs)


def simplify_entities(result: dict[str, Any]) -> dict[str, list[str]]:
    source_entities = result.get("entities") or {}
    entities: dict[str, list[str]] = {}

    for entity_type, values in source_entities.items():
        extracted: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not isinstance(value, dict):
                continue
            text = value.get("text")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            normalized = " ".join(text.split()).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            extracted.append(text)

        if extracted:
            entities[entity_type] = extracted

    return entities


def entity_clause(entity_type: str, value: str) -> dict[str, Any]:
    """Entidades GLiNER funcionam como sinais de relevância, não filtros obrigatórios."""
    return {
        "match": {
            f"entities.{entity_type}": {
                "query": value,
                "operator": "and",
                "boost": ENTITY_BOOSTS.get(entity_type, 1.0),
            }
        }
    }


def build_query(query_text: str, entities: dict[str, list[str]]) -> dict[str, Any]:
    should = [
        entity_clause(entity_type, value)
        for entity_type, values in entities.items()
        for value in values
    ]

    bool_query: dict[str, Any] = {
        "must": [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": ["identifica^3", "arttype^2", "texto_plain"],
                    "type": "best_fields",
                    "operator": "or",
                    "minimum_should_match": "60%",
                }
            }
        ]
    }

    if should:
        bool_query["should"] = should

    return {"bool": bool_query}


def build_search_body(
    query_text: str, entities: dict[str, list[str]], size: int
) -> dict[str, Any]:
    return {
        "size": size,
        "_source": [
            "id",
            "pubdate",
            "pubname",
            "arttype",
            "artcategory",
            "identifica",
            "entities",
        ],
        "query": build_query(query_text=query_text, entities=entities),
    }


def print_results(response: dict[str, Any]) -> None:
    hits = response.get("hits", {})
    total = hits.get("total", {})
    total_value = total.get("value", 0) if isinstance(total, dict) else total

    print("\n=== Resultados ===")
    print(f"total={total_value}")
    print(f"took={response.get('took')}ms")

    for number, hit in enumerate(hits.get("hits", []), start=1):
        source = hit.get("_source", {})
        print(f"\n[{number}] score={hit.get('_score')}")
        print(f"id={source.get('id')}")
        print(f"data={source.get('pubdate')}")
        print(f"tipo={source.get('arttype')}")
        print(f"identifica={source.get('identifica')}")

        for entity_type, values in (source.get("entities") or {}).items():
            if values:
                print(f"{entity_type}={', '.join(str(v) for v in values)}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("Carregando GLiNER2...")
    extractor = GlinerExtractor.from_config(config)

    print("\n=== Consulta ===")
    print(args.query)

    start = time.perf_counter()
    extraction_result = extractor.extract(args.query)
    extraction_time = time.perf_counter() - start
    entities = simplify_entities(extraction_result)

    print("\n=== Entidades extraídas ===")
    print(json.dumps(entities, ensure_ascii=False, indent=2))
    print(f"\ntempo_gliner={extraction_time:.3f}s")

    body = build_search_body(query_text=args.query, entities=entities, size=args.size)

    if args.show_dsl:
        print("\n=== OpenSearch DSL ===")
        print(json.dumps(body, ensure_ascii=False, indent=2))

    client = create_client(args.host, args.username, args.password)
    start = time.perf_counter()
    response = client.search(index=args.index, body=body)
    search_time = time.perf_counter() - start

    print_results(response)
    print("\n=== Tempos ===")
    print(f"gliner={extraction_time:.3f}s")
    print(f"opensearch={search_time:.3f}s")
    print(f"total={extraction_time + search_time:.3f}s")


if __name__ == "__main__":
    main()
