import argparse, json, statistics, sys, time
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ro_dou_gliner import GlinerExtractor

def iter_documents(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield 1, {"texto_plain": path.read_text(encoding="utf-8")}
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for n, doc in enumerate(data if isinstance(data, list) else [data], 1):
            yield n, doc
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield n, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON inválido na linha {n}: {exc}") from exc
    else:
        raise ValueError("Use .txt, .json ou .jsonl")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--text-field", default="texto_plain")
    p.add_argument("--output", type=Path, default=Path("output/gliner-results.jsonl"))
    p.add_argument("--max-documents", type=int)
    p.add_argument("--start-document", type=int, default=1)
    a = p.parse_args()

    cfg = yaml.safe_load((Path(__file__).parents[1]/"config/gliner.yaml").read_text(encoding="utf-8"))
    print("Carregando GLiNER2...", flush=True)
    extractor = GlinerExtractor.from_config(cfg)
    print("GLiNER2 carregado.", flush=True)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    timings, processed, errors = [], 0, 0
    total_start = time.perf_counter()

    with a.output.open("w", encoding="utf-8") as out:
        for source_number, doc in iter_documents(a.input):
            if source_number < a.start_document:
                continue
            if a.max_documents is not None and processed >= a.max_documents:
                break
            text = str(doc.get(a.text_field, "") or "")
            chunks = extractor.split_text(text) if text else []
            print(f"[{source_number}] chars={len(text):,} chunks={len(chunks)}", flush=True)
            if not text:
                processed += 1
                continue
            start = time.perf_counter()
            try:
                result = extractor.enrich_document(doc, a.text_field)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
                result["_benchmark"] = {"source_document": source_number, "elapsed_seconds": round(elapsed, 6)}
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{source_number}] OK - {elapsed:.3f}s", flush=True)
            except Exception as exc:
                errors += 1
                print(f"[{source_number}] ERRO: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            processed += 1

    print("\n=== Benchmark GLiNER2 ===")
    print(f"documentos_processados={processed}")
    print(f"erros={errors}")
    print(f"tempo_total={time.perf_counter()-total_start:.3f}s")
    if timings:
        print(f"tempo_medio={statistics.mean(timings):.3f}s")
        ordered = sorted(timings)
        print(f"p95_aproximado={ordered[min(len(ordered)-1, int(.95*len(ordered)))]:.3f}s")
    print(f"saida={a.output}")

if __name__ == "__main__":
    main()
