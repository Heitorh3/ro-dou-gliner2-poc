#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from typing import Any

from opensearchpy import OpenSearch, helpers

DEFAULT_INDEX = "rodou-gliner-benchmark"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspeciona as entidades GLiNER2 existentes " "no índice OpenSearch."
        )
    )

    parser.add_argument(
        "--host",
        default="http://localhost:9200",
        help="Endereço do OpenSearch.",
    )

    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Índice. Padrão: {DEFAULT_INDEX}",
    )

    parser.add_argument(
        "--username",
        default=None,
        help="Usuário do OpenSearch.",
    )

    parser.add_argument(
        "--password",
        default=None,
        help="Senha do OpenSearch.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Quantidade de entidades exibidas por tipo.",
    )

    parser.add_argument(
        "--combinations",
        type=int,
        default=20,
        help="Quantidade de combinações exibidas.",
    )

    return parser.parse_args()


def create_client(
    host: str,
    username: str | None,
    password: str | None,
) -> OpenSearch:
    kwargs: dict[str, Any] = {
        "hosts": [host],
    }

    if username and password:
        kwargs["http_auth"] = (
            username,
            password,
        )

    return OpenSearch(**kwargs)


def normalize(value: str) -> str:
    """
    Normalização utilizada somente para contabilização.

    A primeira grafia encontrada continua sendo usada
    na apresentação dos resultados.
    """

    return " ".join(value.split()).casefold()


def inspect_index(
    client: OpenSearch,
    index: str,
) -> tuple[
    int,
    dict[str, Counter],
    dict[str, dict[str, str]],
    Counter,
    dict[tuple[str, str, str, str], set[str]],
]:
    counters = {entity_type: Counter() for entity_type in ENTITY_TYPES}

    display_values = {entity_type: {} for entity_type in ENTITY_TYPES}

    pair_counter = Counter()

    pair_documents: dict[
        tuple[str, str, str, str],
        set[str],
    ] = {}

    document_count = 0

    query = {
        "_source": [
            "id",
            "identifica",
            "entities",
        ],
        "query": {"match_all": {}},
    }

    for hit in helpers.scan(
        client,
        index=index,
        query=query,
        size=500,
    ):
        document_count += 1

        source = hit.get("_source", {})
        document_id = str(source.get("id") or hit.get("_id"))

        entities = source.get("entities") or {}

        document_entities: dict[
            str,
            list[tuple[str, str]],
        ] = {}

        # --------------------------------------------
        # Frequência individual
        # --------------------------------------------

        for entity_type in ENTITY_TYPES:
            values = entities.get(entity_type) or []

            unique_values = {}

            for value in values:
                if not isinstance(value, str):
                    continue

                value = value.strip()

                if not value:
                    continue

                normalized = normalize(value)

                unique_values.setdefault(
                    normalized,
                    value,
                )

                display_values[entity_type].setdefault(
                    normalized,
                    value,
                )

            document_entities[entity_type] = [
                (normalized, original) for normalized, original in unique_values.items()
            ]

            # Conta documentos que possuem a entidade,
            # e não ocorrências repetidas dentro do mesmo
            # documento.
            for normalized in unique_values:
                counters[entity_type][normalized] += 1

        # --------------------------------------------
        # Combinações entre tipos diferentes
        # --------------------------------------------

        populated_types = [
            entity_type
            for entity_type in ENTITY_TYPES
            if document_entities[entity_type]
        ]

        for type_a, type_b in combinations(
            populated_types,
            2,
        ):
            for normalized_a, original_a in document_entities[type_a]:
                for normalized_b, original_b in document_entities[type_b]:
                    key = (
                        type_a,
                        normalized_a,
                        type_b,
                        normalized_b,
                    )

                    pair_counter[key] += 1

                    pair_documents.setdefault(
                        key,
                        set(),
                    ).add(document_id)

    return (
        document_count,
        counters,
        display_values,
        pair_counter,
        pair_documents,
    )


def print_entity_frequencies(
    counters: dict[str, Counter],
    display_values: dict[str, dict[str, str]],
    top: int,
) -> None:
    print()
    print("=== Frequência das entidades ===")

    for entity_type in ENTITY_TYPES:
        counter = counters[entity_type]

        print()
        print(f"--- {entity_type} ---")
        print(f"valores_distintos={len(counter)}")

        if not counter:
            print("(nenhuma entidade)")
            continue

        for normalized, count in counter.most_common(top):
            value = display_values[entity_type].get(
                normalized,
                normalized,
            )

            print(f"{count:4}  {value}")


def print_combinations(
    pair_counter: Counter,
    pair_documents: dict[
        tuple[str, str, str, str],
        set[str],
    ],
    display_values: dict[str, dict[str, str]],
    limit: int,
) -> None:
    print()
    print("=== Combinações de entidades ===")

    for key, count in pair_counter.most_common(limit):
        (
            type_a,
            normalized_a,
            type_b,
            normalized_b,
        ) = key

        value_a = display_values[type_a].get(
            normalized_a,
            normalized_a,
        )

        value_b = display_values[type_b].get(
            normalized_b,
            normalized_b,
        )

        documents = sorted(
            pair_documents.get(
                key,
                set(),
            )
        )

        print()
        print(f"{type_a}: {value_a}")
        print(f"{type_b}: {value_b}")
        print(f"documentos={count}")

        if len(documents) <= 5:
            print(f"ids={', '.join(documents)}")


def main() -> None:
    args = parse_args()

    client = create_client(
        host=args.host,
        username=args.username,
        password=args.password,
    )

    if not client.ping():
        raise RuntimeError(f"Não foi possível conectar ao OpenSearch: " f"{args.host}")

    if not client.indices.exists(index=args.index):
        raise RuntimeError(f"O índice '{args.index}' não existe.")

    print("=== Inspeção do índice ===")
    print(f"host={args.host}")
    print(f"index={args.index}")

    (
        document_count,
        counters,
        display_values,
        pair_counter,
        pair_documents,
    ) = inspect_index(
        client=client,
        index=args.index,
    )

    print(f"documentos={document_count}")

    print_entity_frequencies(
        counters=counters,
        display_values=display_values,
        top=args.top,
    )

    print_combinations(
        pair_counter=pair_counter,
        pair_documents=pair_documents,
        display_values=display_values,
        limit=args.combinations,
    )


if __name__ == "__main__":
    main()
