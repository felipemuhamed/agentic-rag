# Agentic RAG

Implementacao incremental da arquitetura descrita em
`agentic-rag-arquitetura-roadmap-codex.pdf`.

## Estado atual

Etapa 1 (bootstrap da API): aceita. O endpoint de saude nao carrega LangGraph,
Chroma, modelos de embedding ou qualquer servico externo.

Etapa 2 (knowledge ingestion): implementada. Suporta TXT e Markdown,
normalizacao, metadata rastreavel, chunking configuravel, embeddings com
Sentence Transformers e persistencia em um servidor Chroma via HTTP. O modo
persistente embutido permanece disponivel para testes isolados.

Etapa 3 (knowledge retrieval): implementada. Executa dense retrieval no
Chroma, selecao por MMR e reranking com CrossEncoder sem depender de LangGraph.

Os arquivos preexistentes em `app/graph`, `app/llm` e `app/rag` continuam como
rascunhos de etapas posteriores e nao representam etapas aceitas do roadmap.

## Requisitos

- Python 3.11 ou superior

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[knowledge,dev]'
cp .env.example .env
```

## Executar

```bash
uvicorn app.main:app --reload
```

A verificacao de saude fica disponivel em `http://127.0.0.1:8000/health`.

## Executar com Docker Compose

Prepare a configuracao e suba API e Chroma:

```bash
cp .env.example .env
docker compose build api
docker compose up -d api
docker compose ps
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/api/v2/healthcheck
```

Para acompanhar ou encerrar:

```bash
docker compose logs -f api chroma
docker compose down
```

A imagem `api` nao instala o servidor Chroma nem modelos de ML. O Chroma roda
no servico `chroma`, usando a imagem oficial fixada em `1.5.9`. A ingestao usa
o client HTTP presente no target `knowledge`, executado sob demanda:

```bash
cp caminho/guia.md documents/guia.md
docker compose build ingest
docker compose run --rm ingest /documents/guia.md
```

Depois de ingerir os documentos, o retriever pode ser executado isoladamente:

```bash
docker compose run --rm retrieve "Como renovar um access token?"
```

A base do servidor Chroma e o cache dos modelos ficam em volumes nomeados.
`docker compose down` os preserva; `docker compose down -v` tambem os remove e
apaga o conhecimento indexado. Consulte
[`docs/containers.md`](docs/containers.md) para os detalhes.

## Testar

```bash
pytest
```

O plano de entregas e os gates entre iteracoes estao em
[`docs/implementation-roadmap.md`](docs/implementation-roadmap.md).

## Ingerir conhecimento

Com o container Chroma em execucao, o comando abaixo baixa o modelo de
embeddings configurado na primeira execucao e envia os chunks por HTTP:

```bash
docker compose up -d chroma
agentic-rag-ingest docs/guia.md documentos/politica.txt
```

Os principais parametros podem ser sobrescritos por ambiente ou por CLI:

```bash
agentic-rag-ingest documento.md \
  --chroma-mode http \
  --chroma-host localhost \
  --chroma-port 8001 \
  --collection knowledge \
  --embedding-model BAAI/bge-m3 \
  --chunk-size 700 \
  --chunk-overlap 100
```

Para inspecionar a collection programaticamente:

```python
from app.knowledge.vectorstore import ChromaVectorStore

store = ChromaVectorStore.http(
    "knowledge",
    host="localhost",
    port=8001,
)
print(store.count())
print(store.get_document_chunks("<document_id>"))
```

Para um teste sem servidor externo, ainda e possivel usar
`--chroma-mode persistent --chroma-path ./chroma`.

Reingerir o mesmo caminho substitui todos os chunks daquele documento. O
`document_id` depende do caminho absoluto da fonte; os IDs dos chunks dependem
do documento, indice e conteudo normalizado.

## Testar retrieval

Com o Chroma preenchido, execute sem LangGraph ou API:

```bash
agentic-rag-retrieve "Como renovar um access token?"
```

Filtros Chroma podem ser fornecidos como JSON:

```bash
agentic-rag-retrieve \
  "Como renovar um access token?" \
  --where '{"section":"OAuth"}'
```

O resultado inclui os chunks finais, metadata, scores e contagens de cada fase.
Veja [`docs/retrieval.md`](docs/retrieval.md) para o contrato e a estrategia de
calibracao.
