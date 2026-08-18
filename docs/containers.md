# Containers de desenvolvimento

Esta estrutura containeriza apenas os componentes ja aceitos: API, knowledge
ingestion e o servidor Chroma. Ela nao representa o hardening de producao da
Etapa 12.

## Imagens

O `Dockerfile` possui dois targets finais:

| Target | Conteudo | Processo padrao |
|---|---|---|
| `api` | Aplicacao e dependencias HTTP | Uvicorn em `0.0.0.0:8000` |
| `knowledge` | API base, client Chroma e Sentence Transformers | `agentic-rag-ingest` |

Ambos executam com o usuario sem privilegios `app` (`uid=10001`). As wheels sao
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
`/health`.

```bash
docker compose up --build -d api
docker compose ps
```

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

## Persistencia

| Volume | Montagem | Finalidade |
|---|---|---|
| `chroma-data` | Chroma: `/data` | Collection e embeddings persistidos |
| `model-cache` | `/home/app/.cache/huggingface` | Modelos baixados pelo provider |

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
modelo, collection e parametros de chunking. API e Chroma aceitam conexoes do
host apenas em `127.0.0.1` por padrao.

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
docker compose down
```
