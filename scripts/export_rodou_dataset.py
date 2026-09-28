#!/usr/bin/env python3
import argparse
import base64
import json
import os
import random
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "id",
    "pubdate",
    "pubname",
    "arttype",
    "artcategory",
    "identifica",
    "orgao",
    "department",
    "assina",
    "texto_plain",
]


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    user: str | None = None,
    password: str | None = None,
    insecure: bool = False,
) -> Any:
    headers = {"Accept": "application/json"}

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    if user:
        token = base64.b64encode(
            f"{user}:{password or ''}".encode("utf-8")
        ).decode("ascii")
        headers["Authorization"] = f"Basic {token}"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    context = None
    if insecure:
        context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(request, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenSearch retornou HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Não foi possível conectar ao OpenSearch: {exc}"
        ) from exc


def list_indices(host: str, user: str | None, password: str | None, insecure: bool):
    url = host.rstrip("/") + "/_cat/indices?format=json&h=index,docs.count,store.size"
    indices = request_json(
        url,
        user=user,
        password=password,
        insecure=insecure,
    )

    for item in sorted(indices, key=lambda x: x.get("index", "")):
        print(
            f"{item.get('index', ''):45} "
            f"docs={item.get('docs.count', '?'):>10} "
            f"size={item.get('store.size', '?')}"
        )


def build_query(args) -> dict[str, Any]:
    filters: list[dict[str, Any]] = []

    if args.from_date or args.to_date:
        date_range: dict[str, Any] = {}
        if args.from_date:
            date_range["gte"] = args.from_date
        if args.to_date:
            date_range["lte"] = args.to_date
        filters.append({"range": {args.date_field: date_range}})

    if args.sections:
        filters.append(
            {"terms": {args.section_field: args.sections}}
        )

    if args.arttypes:
        filters.append(
            {"terms": {args.arttype_field: args.arttypes}}
        )

    if args.organization:
        filters.append(
            {
                "match_phrase": {
                    args.organization_field: args.organization
                }
            }
        )

    query: dict[str, Any]
    if filters:
        query = {"bool": {"filter": filters}}
    else:
        query = {"match_all": {}}

    return query


def export_documents(args) -> list[dict[str, Any]]:
    host = args.host.rstrip("/")
    search_url = (
        f"{host}/{urllib.parse.quote(args.index, safe='*,-_')}/_search"
    )

    fields = args.fields or DEFAULT_FIELDS
    query = build_query(args)

    documents: list[dict[str, Any]] = []
    search_after = None

    while len(documents) < args.limit:
        size = min(args.page_size, args.limit - len(documents))

        body: dict[str, Any] = {
            "size": size,
            "query": query,
            "_source": fields,
            "sort": [
                {args.date_field: {"order": "asc"}},
                {"_id": {"order": "asc"}},
            ],
        }

        if search_after is not None:
            body["search_after"] = search_after

        response = request_json(
            search_url,
            method="POST",
            body=body,
            user=args.user,
            password=args.password,
            insecure=args.insecure,
        )

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            source = dict(hit.get("_source") or {})

            text = source.get("texto_plain")
            if text is None or not str(text).strip():
                continue

            source.setdefault("_opensearch_id", hit.get("_id"))
            documents.append(source)

            if len(documents) >= args.limit:
                break

        search_after = hits[-1].get("sort")
        if not search_after:
            break

    if args.sample and args.sample < len(documents):
        rng = random.Random(args.seed)
        documents = rng.sample(documents, args.sample)

    return documents


def write_jsonl(path: Path, documents: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output:
        for document in documents:
            json.dump(
                document,
                output,
                ensure_ascii=False,
                default=str,
            )
            output.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exporta publicações já indexadas pelo Ro-DOU/OpenSearch "
            "para um dataset JSONL usado no benchmark GLiNER2."
        )
    )

    parser.add_argument(
        "--host",
        default=os.getenv("OPENSEARCH_HOST", "http://localhost:9200"),
    )
    parser.add_argument(
        "--user",
        default=os.getenv("OPENSEARCH_USER"),
    )
    parser.add_argument(
        "--password",
        default=os.getenv("OPENSEARCH_PASS"),
    )
    parser.add_argument("--index")
    parser.add_argument("--list-indices", action="store_true")

    parser.add_argument("--from-date")
    parser.add_argument("--to-date")

    parser.add_argument("--sections", nargs="+")
    parser.add_argument("--arttypes", nargs="+")
    parser.add_argument("--organization")

    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=250)
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/publicacoes.jsonl"),
    )

    parser.add_argument("--insecure", action="store_true")

    parser.add_argument("--date-field", default="pubdate")
    parser.add_argument("--section-field", default="pubname")
    parser.add_argument("--arttype-field", default="arttype")
    parser.add_argument("--organization-field", default="orgao")
    parser.add_argument("--fields", nargs="+")

    args = parser.parse_args()

    if args.list_indices:
        list_indices(
            args.host,
            args.user,
            args.password,
            args.insecure,
        )
        return

    if not args.index:
        parser.error("--index é obrigatório, exceto com --list-indices")

    documents = export_documents(args)
    write_jsonl(args.output, documents)

    print(f"documentos_exportados={len(documents)}")
    print(f"saida={args.output}")


if __name__ == "__main__":
    main()
