#!/usr/bin/env python3

"""
Indexa documentos JSONL preparados pelo prepare_index.py no OpenSearch.

Exemplo:

    uv run python index_documents.py \
        output/opensearch-documents.jsonl \
        --host http://localhost:9200 \
        --index rodou-gliner
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch, helpers

DEFAULT_INDEX = "rodou-gliner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indexa documentos preparados no OpenSearch."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Arquivo JSONL gerado pelo prepare_index.py.",
    )

    parser.add_argument(
        "--host",
        default="http://localhost:9200",
        help="Endereço do OpenSearch.",
    )

    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"Índice de destino. Padrão: {DEFAULT_INDEX}",
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
        "--batch-size",
        type=int,
        default=100,
        help="Quantidade de documentos por lote.",
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
        kwargs["http_auth"] = (username, password)

    return OpenSearch(**kwargs)


def generate_actions(
    input_path: Path,
    index_name: str,
):
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[ERRO] JSON inválido na linha " f"{line_number}: {exc}")
                continue

            document_id = document.get("_opensearch_id") or document.get("id")

            action = {
                "_index": index_name,
                "_source": document,
            }

            if document_id is not None:
                action["_id"] = str(document_id)

            yield action


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.input}")

    client = create_client(
        host=args.host,
        username=args.username,
        password=args.password,
    )

    if not client.ping():
        raise RuntimeError(f"Não foi possível conectar ao OpenSearch: {args.host}")

    if not client.indices.exists(index=args.index):
        raise RuntimeError(f"O índice '{args.index}' não existe.")

    print(f"OpenSearch: {args.host}")
    print(f"Índice:     {args.index}")
    print(f"Arquivo:    {args.input}")
    print()

    success, errors = helpers.bulk(
        client,
        generate_actions(
            input_path=args.input,
            index_name=args.index,
        ),
        chunk_size=args.batch_size,
        raise_on_error=False,
        raise_on_exception=False,
    )

    client.indices.refresh(index=args.index)

    count = client.count(index=args.index)["count"]

    print()
    print("=== Indexação concluída ===")
    print(f"documentos_indexados={success}")
    print(f"erros={len(errors)}")
    print(f"documentos_no_indice={count}")

    if errors:
        print()
        print("=== Primeiros erros ===")

        for error in errors[:10]:
            print(
                json.dumps(
                    error,
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
