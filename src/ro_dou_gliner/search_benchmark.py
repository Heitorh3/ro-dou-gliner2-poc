#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

import yaml
from opensearchpy import OpenSearch

from .extractor import GlinerExtractor

DEFAULT_INDEX = "rodou-gliner-benchmark"

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

# Consultas baseadas em entidades que sabemos existir no corpus atual.
# expected_entities define o conjunto de relevância: um documento é considerado
# relevante quando contém TODAS as entidades esperadas, nos respectivos tipos.
BENCHMARK_CASES = [
    {
        "name": "anm_alvara",
        "query": "publicações da Agência Nacional de Mineração sobre Alvará de Pesquisa",
        "expected_entities": {
            "organization": ["AGÊNCIA NACIONAL DE MINERAÇÃO - ANM"],
            "legal_act": ["Alvará de Pesquisa"],
        },
    },
    {
        "name": "anm_codigo_mineracao",
        "query": "atos da Agência Nacional de Mineração relacionados ao Código de Mineração",
        "expected_entities": {
            "organization": ["AGÊNCIA NACIONAL DE MINERAÇÃO - ANM"],
            "legal_act": ["Código de Mineração"],
        },
    },
    {
        "name": "janduci_alvara",
        "query": "publicações de Janduci Dutra Fernandes sobre Alvará de Pesquisa",
        "expected_entities": {
            "person": ["Janduci Dutra Fernandes"],
            "legal_act": ["Alvará de Pesquisa"],
        },
    },
    {
        "name": "janduci_codigo_mineracao",
        "query": "atos de Janduci Dutra Fernandes relacionados ao Código de Mineração",
        "expected_entities": {
            "person": ["Janduci Dutra Fernandes"],
            "legal_act": ["Código de Mineração"],
        },
    },
    {
        "name": "anm_coordenadora_tah",
        "query": "publicações da Agência Nacional de Mineração da Coordenadora de Autuação e Fiscalização da TAH",
        "expected_entities": {
            "organization": ["AGÊNCIA NACIONAL DE MINERAÇÃO - ANM"],
            "position": ["Coordenadora de Autuação e Fiscalização da TAH"],
        },
    },
]


@dataclass
class SearchMetrics:
    total_hits: int
    relevant_total: int
    relevant_at_k: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    search_ms: float
    top_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara busca textual com busca textual enriquecida por GLiNER2."
    )
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--config", default="config/gliner.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--show-results", action="store_true")
    parser.add_argument(
        "--cases",
        default=None,
        help="JSON com casos adicionais/substitutos. Se informado, deve ser uma lista.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_cases(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return BENCHMARK_CASES
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("O arquivo --cases deve conter uma lista JSON.")
    return data


def create_client(host: str, username: str | None, password: str | None) -> OpenSearch:
    kwargs: dict[str, Any] = {"hosts": [host]}
    if username and password:
        kwargs["http_auth"] = (username, password)
    return OpenSearch(**kwargs)


def normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


def simplify_entities(result: dict[str, Any]) -> dict[str, list[str]]:
    source = result.get("entities") or {}
    output: dict[str, list[str]] = {}
    for entity_type, values in source.items():
        seen: set[str] = set()
        extracted: list[str] = []
        for item in values or []:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            key = normalize(text)
            if key in seen:
                continue
            seen.add(key)
            extracted.append(text)
        if extracted:
            output[entity_type] = extracted
    return output


def textual_query(query_text: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": query_text,
            "fields": ["identifica^3", "arttype^2", "texto_plain"],
            "type": "best_fields",
            "operator": "or",
            "minimum_should_match": "60%",
        }
    }


def hybrid_query(query_text: str, entities: dict[str, list[str]]) -> dict[str, Any]:
    should: list[dict[str, Any]] = []
    for entity_type, values in entities.items():
        boost = ENTITY_BOOSTS.get(entity_type, 1.0)
        for value in values:
            should.append(
                {
                    "match": {
                        f"entities.{entity_type}": {
                            "query": value,
                            "operator": "and",
                            "boost": boost,
                        }
                    }
                }
            )

    query: dict[str, Any] = {
        "bool": {
            "must": [textual_query(query_text)],
        }
    }
    if should:
        query["bool"]["should"] = should
    return query


def expected_filter(expected_entities: dict[str, list[str]]) -> dict[str, Any]:
    """Cria um filtro independente da query testada para formar o gold set.

    Usa match/operator=and nos campos de entidades. Para este benchmark inicial,
    os casos foram escolhidos a partir das entidades existentes no próprio índice.
    """
    must: list[dict[str, Any]] = []
    for entity_type, values in expected_entities.items():
        for value in values:
            must.append(
                {
                    "match": {
                        f"entities.{entity_type}": {
                            "query": value,
                            "operator": "and",
                        }
                    }
                }
            )
    return {"bool": {"must": must}}


def get_relevant_ids(
    client: OpenSearch,
    index: str,
    expected_entities: dict[str, list[str]],
) -> set[str]:
    response = client.search(
        index=index,
        body={
            "size": 10000,
            "_source": False,
            "query": expected_filter(expected_entities),
        },
    )
    return {str(hit["_id"]) for hit in response["hits"]["hits"]}


def dcg(relevance: list[int]) -> float:
    import math

    score = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            score += 1.0 / math.log2(rank + 1)
    return score


def evaluate(
    response: dict[str, Any],
    relevant_ids: set[str],
    k: int,
    elapsed_ms: float,
) -> SearchMetrics:
    hits = response.get("hits", {}).get("hits", [])[:k]
    ids = [str(hit.get("_id")) for hit in hits]
    relevance = [1 if doc_id in relevant_ids else 0 for doc_id in ids]

    relevant_at_k = sum(relevance)
    precision = relevant_at_k / len(ids) if ids else 0.0
    recall = relevant_at_k / len(relevant_ids) if relevant_ids else 0.0

    rr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            rr = 1.0 / rank
            break

    ideal = [1] * min(len(relevant_ids), k)
    ideal_dcg = dcg(ideal)
    ndcg = dcg(relevance) / ideal_dcg if ideal_dcg else 0.0

    total = response.get("hits", {}).get("total", 0)
    if isinstance(total, dict):
        total = int(total.get("value", 0))

    return SearchMetrics(
        total_hits=int(total),
        relevant_total=len(relevant_ids),
        relevant_at_k=relevant_at_k,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        search_ms=elapsed_ms,
        top_ids=ids,
    )


def run_search(
    client: OpenSearch,
    index: str,
    query: dict[str, Any],
    k: int,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = client.search(
        index=index,
        body={
            "size": k,
            "_source": ["id", "identifica", "arttype", "entities"],
            "query": query,
        },
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def fmt(value: float) -> str:
    return f"{value:.3f}"


def print_metrics(label: str, metrics: SearchMetrics, k: int) -> None:
    print(
        f"{label:9} "
        f"hits={metrics.total_hits:<4} "
        f"rel@{k}={metrics.relevant_at_k:<2}/{metrics.relevant_total:<3} "
        f"P@{k}={fmt(metrics.precision_at_k)} "
        f"R@{k}={fmt(metrics.recall_at_k)} "
        f"MRR={fmt(metrics.reciprocal_rank)} "
        f"nDCG@{k}={fmt(metrics.ndcg_at_k)} "
        f"time={metrics.search_ms:.1f}ms"
    )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    config = load_config(args.config)

    client = create_client(args.host, args.username, args.password)
    if not client.ping():
        raise RuntimeError(f"Não foi possível conectar ao OpenSearch: {args.host}")
    if not client.indices.exists(index=args.index):
        raise RuntimeError(f"O índice '{args.index}' não existe.")

    print("Carregando GLiNER2...")
    extractor = GlinerExtractor.from_config(config)

    textual_metrics: list[SearchMetrics] = []
    hybrid_metrics: list[SearchMetrics] = []
    gliner_times: list[float] = []

    print()
    print(f"=== Benchmark de busca (k={args.k}) ===")

    for number, case in enumerate(cases, start=1):
        query_text = case["query"]
        expected = case["expected_entities"]

        relevant_ids = get_relevant_ids(client, args.index, expected)

        start = time.perf_counter()
        extraction = extractor.extract(query_text)
        gliner_ms = (time.perf_counter() - start) * 1000
        gliner_times.append(gliner_ms)
        entities = simplify_entities(extraction)

        textual_response, textual_ms = run_search(
            client, args.index, textual_query(query_text), args.k
        )
        hybrid_response, hybrid_ms = run_search(
            client, args.index, hybrid_query(query_text, entities), args.k
        )

        text_m = evaluate(textual_response, relevant_ids, args.k, textual_ms)
        hybrid_m = evaluate(
            hybrid_response, relevant_ids, args.k, hybrid_ms + gliner_ms
        )
        textual_metrics.append(text_m)
        hybrid_metrics.append(hybrid_m)

        print()
        print(f"[{number}] {case.get('name', 'case')}")
        print(f"consulta={query_text}")
        print("entidades_gliner=" + json.dumps(entities, ensure_ascii=False))
        print(f"gold_documents={len(relevant_ids)} gliner={gliner_ms:.1f}ms")
        print_metrics("textual", text_m, args.k)
        print_metrics("hybrid", hybrid_m, args.k)

        if args.show_results:
            print(f"textual_top_ids={text_m.top_ids}")
            print(f"hybrid_top_ids={hybrid_m.top_ids}")

    if not cases:
        return

    print()
    print("=== Resumo ===")
    print(f"consultas={len(cases)}")
    print(f"gliner_medio={mean(gliner_times):.1f}ms")
    print()
    print(
        f"textual: "
        f"P@{args.k}={mean(m.precision_at_k for m in textual_metrics):.3f} "
        f"R@{args.k}={mean(m.recall_at_k for m in textual_metrics):.3f} "
        f"MRR={mean(m.reciprocal_rank for m in textual_metrics):.3f} "
        f"nDCG@{args.k}={mean(m.ndcg_at_k for m in textual_metrics):.3f} "
        f"tempo={mean(m.search_ms for m in textual_metrics):.1f}ms"
    )
    print(
        f"hybrid:  "
        f"P@{args.k}={mean(m.precision_at_k for m in hybrid_metrics):.3f} "
        f"R@{args.k}={mean(m.recall_at_k for m in hybrid_metrics):.3f} "
        f"MRR={mean(m.reciprocal_rank for m in hybrid_metrics):.3f} "
        f"nDCG@{args.k}={mean(m.ndcg_at_k for m in hybrid_metrics):.3f} "
        f"tempo={mean(m.search_ms for m in hybrid_metrics):.1f}ms"
    )

    print()
    print("Observação: o gold set deste benchmark é derivado das entidades já")
    print("indexadas. Portanto, ele mede principalmente o efeito do GLiNER no")
    print("ranking das entidades conhecidas; não substitui avaliação humana.")


if __name__ == "__main__":
    main()
