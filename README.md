# Ro-DOU + GLiNER2 POC v5

A v5 trata publicações longas com chunking adaptativo.

- streaming de JSONL;
- saída incremental;
- `--start-document` e `--max-documents`;
- log de caracteres/chunks;
- chunks de 6.000 caracteres;
- overlap de 500 caracteres;
- preferência por corte em quebra de linha;
- ajuste de offsets;
- merge/deduplicação dos resultados;
- `max_len: null`, evitando truncamento deliberado.

## Teste

```bash
python scripts/benchmark.py dataset/publicacoes.jsonl   --text-field texto_plain   --max-documents 10   --output output/gliner-results.jsonl
```

## Linha 416 (outlier de 51.826 caracteres)

```bash
python scripts/benchmark.py dataset/publicacoes.jsonl   --start-document 416   --max-documents 1   --text-field texto_plain   --output output/doc-416.jsonl
```


## Importar publicações reais do Ro-DOU/OpenSearch

A v5 mantém o exportador da v4 para criar o dataset real usado no benchmark.

Configure a conexão:

```bash
export OPENSEARCH_HOST=http://localhost:9200
export OPENSEARCH_USER='...'
export OPENSEARCH_PASS='...'
```

Liste os índices:

```bash
python scripts/export_rodou_dataset.py --list-indices
```

Exporte até 500 publicações:

```bash
python scripts/export_rodou_dataset.py   --index NOME_DO_INDICE   --from-date 2026-09-01   --to-date 2026-09-18   --limit 500   --output dataset/publicacoes.jsonl
```

Depois execute:

```bash
python scripts/benchmark.py dataset/publicacoes.jsonl   --text-field texto_plain   --output output/gliner-results.jsonl
```

Exemplo:

```bash
uv run python -m ro_dou_gliner.search "publicações da Agência Nacional de Mineração sobre Alvará de Pesquisa"
```