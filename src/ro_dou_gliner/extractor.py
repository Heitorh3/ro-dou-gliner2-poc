from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from gliner2 import AutoExtractor

DEFAULT_ENTITIES = {    
    "person": (
        "Nome completo de pessoa física, incluindo servidor, "
        "autoridade, beneficiário ou signatário."
    ),

    "organization": (
        "Nome de órgão público, entidade, universidade, empresa, "
        "departamento, secretaria, coordenação ou unidade administrativa."
    ),

    "position": (
        "Cargo, emprego público, função, função comissionada ou "
        "função gratificada ocupada por uma pessoa."
    ),

    "location": (
        "Município, estado, país ou outra localização geográfica."
    ),

    "process": (
        "Número identificador de processo administrativo ou judicial, "
        "incluindo Processo SEI, PAe e formatos equivalentes."
    ),

    "legal_act": (
        "Identificação de ato normativo ou administrativo, incluindo "
        "Portaria, Decreto, Resolução, Edital, Lei ou Emenda Constitucional."
    ),

    "registration": (
        "Número de matrícula ou identificação funcional de servidor, "
        "incluindo matrícula SIAPE."
    ),
}

@dataclass(frozen=True)
class GlinerConfig:
    enabled: bool = False
    model: str = "fastino/gliner2.5-multi-v1"
    threshold: float = 0.45
    max_len: int | None = None
    include_confidence: bool = True
    include_spans: bool = True
    chunking_enabled: bool = True
    chunk_size_chars: int = 6000
    chunk_overlap_chars: int = 500

class GlinerExtractor:
    def __init__(self, config: GlinerConfig, schema_config: Mapping[str, Any]):
        self.config = config
        self.schema_config = dict(schema_config)
        self.model = AutoExtractor.from_pretrained(config.model) if config.enabled else None
        self.schema = self._build_schema() if self.model is not None else None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GlinerExtractor:
        s = config.get("gliner", config)
        c = s.get("chunking") or {}
        max_len = s.get("max_len")
        return cls(
            GlinerConfig(
                enabled=bool(s.get("enabled", False)),
                model=str(s.get("model", "fastino/gliner2.5-multi-v1")),
                threshold=float(s.get("threshold", 0.45)),
                max_len=int(max_len) if max_len is not None else None,
                include_confidence=bool(s.get("include_confidence", True)),
                include_spans=bool(s.get("include_spans", True)),
                chunking_enabled=bool(c.get("enabled", True)),
                chunk_size_chars=int(c.get("size_chars", 6000)),
                chunk_overlap_chars=int(c.get("overlap_chars", 500)),
            ),
            {
                "entities": s.get("entities") or DEFAULT_ENTITIES,
                "classification": s.get("classification"),
                "relations": s.get("relations"),
            },
        )

    def _build_schema(self):
        schema = self.model.create_schema()
        if self.schema_config.get("entities"):
            schema.entities(self.schema_config["entities"])
        cls = self.schema_config.get("classification")
        if cls and cls.get("enabled", False):
            schema.classification(
                task=cls["task"], labels=cls["labels"],
                multi_label=bool(cls.get("multi_label", False)),
                cls_threshold=float(cls.get("threshold", 0.5)),
            )
        rel = self.schema_config.get("relations")
        if rel and rel.get("enabled", False) and rel.get("types"):
            schema.relations(rel["types"], threshold=rel.get("threshold"))
        return schema

    def schema_as_dict(self):
        return self.schema.to_dict() if self.schema is not None else {}

    def split_text(self, text: str) -> list[tuple[int, int, str]]:
        size, overlap = self.config.chunk_size_chars, self.config.chunk_overlap_chars
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("Configuração inválida de chunking.")
        if len(text) <= size:
            return [(0, len(text), text)]
        chunks, start = [], 0
        while start < len(text):
            target_end = min(start + size, len(text))
            end = target_end
            if target_end < len(text):
                newline = text.rfind("\n", start + int(size * .70), target_end)
                if newline > start:
                    end = newline + 1
            chunks.append((start, end, text[start:end]))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _extract_single(self, text: str):
        if self.model is None or self.schema is None:
            raise RuntimeError("GLiNER2 habilitado sem modelo/schema carregado.")
        return self.model.extract(
            text, self.schema, threshold=self.config.threshold,
            include_confidence=self.config.include_confidence,
            include_spans=self.config.include_spans,
            max_len=self.config.max_len,
        )

    @staticmethod
    def _offset_spans(value: Any, offset: int) -> Any:
        if isinstance(value, list):
            return [GlinerExtractor._offset_spans(v, offset) for v in value]
        if not isinstance(value, dict):
            return value
        out = {k: GlinerExtractor._offset_spans(v, offset) for k, v in value.items()}
        for a, b in (("start", "end"), ("start_char", "end_char"), ("span_start", "span_end")):
            if isinstance(value.get(a), int) and isinstance(value.get(b), int):
                out[a], out[b] = value[a] + offset, value[b] + offset
        span = value.get("span")
        if isinstance(span, (list, tuple)) and len(span) == 2 and all(isinstance(x, int) for x in span):
            out["span"] = [span[0] + offset, span[1] + offset]
        return out

    @staticmethod
    def _dedupe(items):
        seen, out = set(), []
        for item in items:
            key = repr(item)
            if key not in seen:
                seen.add(key); out.append(item)
        return out

    def _merge_results(self, results):
        merged = {}
        for result in results:
            for key, value in result.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(value)
                elif isinstance(value, dict):
                    target = merged.setdefault(key, {})
                    for sk, sv in value.items():
                        if isinstance(sv, list):
                            target.setdefault(sk, []).extend(sv)
                        elif sk not in target:
                            target[sk] = sv
                elif key not in merged:
                    merged[key] = value
        for key, value in merged.items():
            if isinstance(value, list):
                merged[key] = self._dedupe(value)
            elif isinstance(value, dict):
                for sk, sv in value.items():
                    if isinstance(sv, list):
                        value[sk] = self._dedupe(sv)
        return merged

    def extract(self, text: str):
        if not self.config.enabled or not text:
            return {}
        if not self.config.chunking_enabled or len(text) <= self.config.chunk_size_chars:
            return self._extract_single(text)
        results = []
        for start, _, chunk in self.split_text(text):
            results.append(self._offset_spans(self._extract_single(chunk), start))
        return self._merge_results(results)

    def enrich_document(self, document: Mapping[str, Any], text_field="texto_plain"):
        enriched = dict(document)
        text = str(document.get(text_field, "") or "")
        chunks = self.split_text(text) if self.config.chunking_enabled and text else []
        enriched["gliner"] = {
            "model": self.config.model,
            "schema": self.schema_as_dict(),
            "processing": {
                "text_chars": len(text),
                "chunk_count": len(chunks),
                "chunk_size_chars": self.config.chunk_size_chars,
                "chunk_overlap_chars": self.config.chunk_overlap_chars,
                "max_len": self.config.max_len,
            },
            "result": self.extract(text),
        }
        return enriched
