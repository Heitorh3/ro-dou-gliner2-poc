
import json

with open("output/gliner-results.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        doc = json.loads(line)

        print("=" * 80)
        print(f"DOCUMENTO {i}")
        print("Identificação:", doc.get("identifica"))
        print("Órgão:", doc.get("orgao"))

        gliner = doc.get("gliner", {})
        print("Processamento:", gliner.get("processing"))
        print("Resultado:")
        print(json.dumps(
            gliner.get("result"),
            ensure_ascii=False,
            indent=2
        ))
