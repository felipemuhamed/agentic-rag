# Containers e Docker Compose

A estrutura containeriza API, ingestao/retrieval, RAG e Chroma. A Etapa 12
acrescentou persistencia dos threads e reducao de privilegios ao Compose
iniciado durante o desenvolvimento.

## Imagens

O `Dockerfile` possui tres targets finais:

| Target | Conteudo | Processo padrao |
|---|---|---|
| `api` | API HTTP, runtime RAG, web search e adapter do provider | Uvicorn em `0.0.0.0:8000` |
| `knowledge` | API base, client Chroma e Sentence Transformers | `agentic-rag-ingest` |
| `rag` | Knowledge, LangGraph, graders, gerador e web search | `agentic-rag-ask` |

Todos executam como usuario sem privilegios `app` (`uid=10001`). As wheels sao
construidas em stages separados e instaladas nas imagens finais sem ferramentas
de build.

## Servicos Compose

```text
host:8000 -> api -> checkpoint-data, upload-data
                \
                 -> chroma:8000 -> chroma-data
                /
ingest/retrieve/rag

host:8001 -> chroma:8000 (inspecao local)
```

### `chroma`

Executa `chromadb/chroma:1.5.9` em modo servidor e persiste dados em `/data`.
Na rede Compose, os clientes usam `http://chroma:8000`; no host, o endpoint fica
em `http://127.0.0.1:8001` por padrao.

O auxiliar `chroma-ready` consulta `/api/v2/healthcheck` e termina com sucesso
antes que API ou ferramentas sejam liberadas. Ele existe porque a imagem do
Chroma nao fornece as ferramentas usadas pelo healthcheck da API.

### `api`

Sobe junto com o Chroma e usa `/ready` no healthcheck. O grafo e os modelos sao
construidos somente no primeiro chat. O filesystem raiz e somente leitura;
`/tmp` usa tmpfs; todas as capabilities sao removidas; ganho de privilegio e
bloqueado; init e shutdown gracioso evitam processos e checkpoints abandonados.

```bash
docker compose up --build -d api
docker compose ps
```

O servico compartilha `model-cache`, grava o SQLite em `checkpoint-data` e os
arquivos recebidos pelos endpoints de upload em `upload-data`. Para
chat autenticado, configure `LLM_MODEL`, credenciais do provider,
`AUTH_ENABLED`, `API_KEY_POLICIES` e o tenant no `.env`:

```bash
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-a-real-secret' \
  -d '{"conversationId":"demo-1","tenantId":"public","message":"Como renovar o token?"}'
```

O fallback Tavily e opcional e desativado por padrao. Para utiliza-lo,
configure `WEB_SEARCH_ENABLED=true` e `TAVILY_API_KEY`. Com a busca externa
desativada, evidencia interna insuficiente termina em `source=none`, sem
interromper o stream.

### Ferramentas `ingest`, `retrieve` e `rag`

Os tres servicos pertencem ao profile `tools` e nao sobem com a API. Arquivos
para ingestao ficam em `./documents`, montado como somente leitura:

```bash
docker compose run --rm ingest --tenant-id public /documents/arquivo.md
docker compose run --rm retrieve "Como renovar um access token?"
docker compose run --rm rag "Como renovar um access token?"
```

`api` e `rag` instalam `langchain-openrouter==0.2.8` por padrao. Para outro
provider, defina `LLM_PROVIDER_PACKAGE` antes do build e use um `LLM_MODEL`
compativel. O `.env` completo e repassado somente a esses servicos; as
credenciais nao entram na imagem.

## Persistencia

| Volume | Montagem | Finalidade |
|---|---|---|
| `chroma-data` | Chroma: `/data` | Collection e embeddings |
| `checkpoint-data` | API: `/data/checkpoints` | Threads SQLite |
| `upload-data` | API: `/data/uploads` | Arquivos enviados por `POST /documents` |
| `model-cache` | `/home/app/.cache/huggingface` | Modelos baixados |

Chroma armazena apenas a base de conhecimento; mensagens ficam no
checkpointer e os documentos enviados pela API ficam em `upload-data`, que os
mantem disponiveis para reindexacao. Parar os containers preserva todos os volumes:

```bash
docker compose down
```

`docker compose down -v` remove conhecimento, threads e cache e e uma operacao
destrutiva. Defina backup e retencao antes de usar volumes de producao.

## Configuracao e limites

O Compose le `.env` e aplica defaults documentados em `.env.example`. API e
Chroma aceitam conexoes do host apenas em `127.0.0.1` por padrao. O Chroma nao
possui autenticacao nesta topologia e precisa permanecer em rede privada.

`CHECKPOINTER_MODE=sqlite` e apropriado para uma instancia. O rate limiter
tambem e local ao processo. Para replicas concorrentes, use checkpointer e
limites compartilhados. TLS, WAF, secrets e autenticacao entre servicos devem
ser fornecidos pela plataforma. Consulte [`production.md`](production.md).

O Compose usa CPU. Aceleracao requer configuracao do runtime do host e nao
pressupoe NVIDIA Container Toolkit.

## Verificacoes recomendadas

```bash
docker compose config --quiet
docker compose build api
docker compose up -d api
curl --fail http://127.0.0.1:8001/api/v2/healthcheck
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
docker compose run --rm ingest --help
docker compose run --rm retrieve --help
docker compose run --rm rag --help
docker compose down
```
