import json

results = []

with open("output/gliner-results.jsonl", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        benchmark = doc.get("_benchmark", {})
        processing = doc.get("gliner", {}).get("processing", {})

        results.append({
            "doc": benchmark.get("source_document"),
            "chars": processing.get("text_chars"),
            "chunks": processing.get("chunk_count"),
            "tempo": benchmark.get("elapsed_seconds"),
        })

for r in sorted(
    results,
    key=lambda x: x["tempo"] or 0,
    reverse=True,
):
    print(
        f"doc={r['doc']:3} "
        f"chars={r['chars']:6} "
        f"chunks={r['chunks']:2} "
        f"tempo={r['tempo']:.3f}s"
    )