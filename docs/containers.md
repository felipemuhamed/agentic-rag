# Containers de desenvolvimento

Esta estrutura containeriza os componentes implementados: API, knowledge
ingestion/retrieval, RAG com fallback web e o servidor Chroma. Ela nao representa o
hardening de producao da Etapa 12.

## Imagens

O `Dockerfile` possui tres targets finais:

| Target | Conteudo | Processo padrao |
|---|---|---|
| `api` | API HTTP, runtime RAG, web search e adapters de provider | Uvicorn em `0.0.0.0:8000` |
| `knowledge` | API base, client Chroma e Sentence Transformers | `agentic-rag-ingest` |
| `rag` | Knowledge, LangGraph, graders, gerador e web search | `agentic-rag-ask` |

Todos executam com o usuario sem privilegios `app` (`uid=10001`). As wheels sao
construidas em stages separados e instaladas nas imagens finais sem ferramentas
de build.

## Servicos Compose

```text
host:8000 -> api
                \
                 -> chroma:8000 -> chroma-data
                /
ingest --------
retrieve ------
rag -----------

host:8001 -> chroma:8000 (inspecao/desenvolvimento)
```

### `chroma`

Executa `chromadb/chroma:1.5.9` em modo servidor e persiste seus dados em
`/data`. Na rede Compose, os clientes usam `http://chroma:8000`; no host, o
endpoint fica em `http://127.0.0.1:8001` por padrao.

O servico auxiliar `chroma-ready` consulta `/api/v2/healthcheck` e termina com
sucesso antes que API ou ingestao sejam liberadas. Ele e necessario porque a
imagem oficial atual contem o binario Rust, sem interpretador Python para um
healthcheck interno.

### `api`

Sobe por padrao junto com o Chroma, publica `APP_PORT` e possui healthcheck em
`/health`. A imagem inclui knowledge, LangGraph e o adapter do LLM para servir
`POST /chat/stream`, mas o grafo e os modelos so sao construidos na primeira
execucao de chat. Assim, o health check permanece independente do LLM.

```bash
docker compose up --build -d api
docker compose ps
```

O servico recebe a configuracao RAG completa e compartilha `model-cache`. Para
usar chat, configure `LLM_MODEL` e as credenciais do provider no `.env`:

```bash
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"conversationId":"demo-1","message":"Como renovar o token?"}'
```

Configure tambem `TAVILY_API_KEY`. Ela nao e usada enquanto a evidencia interna
for suficiente; sem a chave, apenas uma execucao que alcance o fallback termina
com `error` publico.

O runtime usa um checkpointer em memoria dentro do processo `api`. Repetir
`conversationId` preserva o historico enquanto esse processo estiver vivo, mas
reiniciar/recriar o container apaga as conversas. Replicas distintas tambem nao
compartilham estado nesta fase.

### `ingest`

Pertence ao profile `tools`, portanto nao sobe junto com a API. Ao selecionar o
servico explicitamente, o Compose ativa o profile para aquela execucao:

```bash
docker compose run --rm ingest /documents/arquivo.md
```

Os arquivos de entrada devem estar em `./documents`, montado como somente
leitura em `/documents`.

### `retrieve`

Usa a mesma imagem `knowledge`, substituindo o entrypoint pela CLI de retrieval.
Conecta ao Chroma via HTTP e compartilha o cache dos modelos:

```bash
docker compose run --rm retrieve "Como renovar um access token?"
```

### `rag`

Executa o RAG ate a Etapa 9 e pertence ao profile `tools`. Configure no `.env`
o `LLM_MODEL` no formato `provider:model`, as credenciais do LLM e
`TAVILY_API_KEY`:

```bash
docker compose build rag
docker compose run --rm rag "Como renovar um access token?"
```

O target instala `langchain-openai==1.5.1` por padrao. Para outro provider,
defina `LLM_PROVIDER_PACKAGE` com o pacote de integracao correspondente antes
do build e use um `LLM_MODEL` compativel. O `.env` completo e repassado apenas
ao servico `rag`, permitindo fornecer as variaveis exigidas pelo adapter.
`MAX_RAG_ATTEMPTS` define o total de retrievals, incluindo a consulta original,
e tem padrao `2`. `WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_DEPTH` e
`WEB_SEARCH_TIMEOUT_SECONDS` controlam a unica chamada externa posterior.

## Persistencia

| Volume | Montagem | Finalidade |
|---|---|---|
| `chroma-data` | Chroma: `/data` | Collection e embeddings persistidos |
| `model-cache` | `/home/app/.cache/huggingface` | Modelos baixados pelo provider |

O historico conversacional nao esta nesses volumes. Chroma armazena somente a
base de conhecimento; ele nao e usado como checkpointer do LangGraph.

Parar containers nao remove esses dados:

```bash
docker compose down
```

Para reinicializar completamente o ambiente, inclusive a base indexada e o
cache de modelos:

```bash
docker compose down -v
```

Esse ultimo comando e destrutivo para os dados dos volumes.

## Configuracao

O Compose le `.env` automaticamente. As variaveis documentadas em
`.env.example` controlam portas, binds, tag, versao do Python, imagem Chroma,
modelo, adapter do LLM, collection e parametros de chunking. API e Chroma
aceitam conexoes do host apenas em `127.0.0.1` por padrao. A URL do endpoint
Tavily pode ser substituida por `TAVILY_BASE_URL` para testes ou gateways
aprovados.

O Chroma desta configuracao de desenvolvimento nao possui autenticacao. Nao
publique sua porta em uma interface externa sem adicionar os controles da etapa
de hardening.

Para usar aceleracao, a configuracao de GPU devera ser acrescentada de acordo
com o runtime do host. O Compose atual usa CPU por padrao e nao pressupoe NVIDIA
Container Toolkit.

## Verificacoes recomendadas

Em uma maquina com Docker Compose V2:

```bash
docker compose config --quiet
docker compose build api
docker compose up -d api
curl --fail http://127.0.0.1:8001/api/v2/healthcheck
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').status)"
docker compose run --rm ingest --help
docker compose run --rm retrieve --help
docker compose run --rm rag --help
docker compose down
```
