# Agentic RAG

Implementacao incremental da arquitetura descrita em
`agentic-rag-arquitetura-roadmap-codex.pdf`.

## Estado atual

Etapa 1 (bootstrap da API): aceita. O endpoint de saude nao carrega LangGraph,
Chroma, modelos de embedding ou qualquer servico externo.

Etapa 2 (knowledge ingestion): implementada. Suporta TXT, Markdown e PDF,
normalizacao, metadata rastreavel, chunking configuravel, embeddings com
Sentence Transformers e persistencia em um servidor Chroma via HTTP. O modo
persistente embutido permanece disponivel para testes isolados.

Etapa 3 (knowledge retrieval): implementada. Executa dense retrieval no
Chroma, selecao por MMR e reranking com CrossEncoder sem depender de LangGraph.

Etapa 4 (RAG basico): implementada. O LangGraph executa o caminho linear
`prepare_query -> retrieve_rag -> generate_answer`, com estado minimo e uma
resposta restrita aos documentos recuperados. Essa base foi estendida pelo
grader e pelo ciclo limitado dos gates seguintes.

Etapa 5 (grader RAG): implementada. Uma decisao estruturada e validada separa
contexto suficiente de contexto insuficiente antes da geracao, impedindo que
evidencia inadequada chegue ao modelo de resposta.

Etapa 6 (rewrite e retry): implementada. Grades negativos podem reescrever a
consulta e repetir retrieval/grader ate `MAX_RAG_ATTEMPTS`. O limite inclui a
consulta original e encerra o ciclo interno sem possibilidade de loop infinito.

Etapa 7 (streaming SSE): implementada. `POST /chat/stream` traduz os eventos do
LangGraph para o protocolo publico `status`, `sources`, `token`, `done` e
`error`, transmitindo tokens somente da resposta final.

Etapa 8 (threads e memoria): implementada. `conversationId` identifica a
conversa; o historico serve apenas para transformar follow-ups em consultas
autocontidas. Na API, a identidade autenticada e o tenant tambem isolam o
thread.

Etapa 9 (fallback web): implementada. Depois de esgotar o RAG interno, o grafo
consulta um `WebSearchProvider` substituivel quando `WEB_SEARCH_ENABLED=true`,
avalia a evidencia externa com um grader proprio e so entao responde com
`source=web` ou termina em `no_answer`.

Etapa 10 (citacoes e proveniencia): implementada. Evidencias RAG e web sao
normalizadas em fontes publicas numeradas; toda resposta grounded precisa citar
IDs existentes, e o SSE envia somente as fontes efetivamente utilizadas.

Etapa 11 (avaliacao reproduzivel): implementada. Um dataset e corpus
versionados alimentam Recall@K, MRR e metricas deterministicas de relevancia,
grounding e citacoes; relatorios registram configuracao, latencia, tokens e
custo para comparacoes controladas.

Etapa 12 (hardening de producao): implementada. Inclui checkpointer SQLite
duravel, timeouts/retries/circuit breaker, cancelamento e deadline de streams,
API keys com isolamento por tenant, rate limit, logs JSON, metricas Prometheus
e container com privilegios reduzidos.

Etapa 13 (consulta multimodal): implementada. Imagens PNG, JPEG, WebP ou GIF
podem acompanhar o prompt. Um modelo com visao extrai elementos observaveis,
OCR e identificadores; essa evidencia alimenta o retrieval e pode ser citada na
resposta sem persistir o Base64 no checkpointer.

Etapa 14 (catalogo visual proprietario): implementada. Fotos de referencia sao
indexadas em uma collection visual separada e vinculadas a manuais pelo
`equipment_id`. Uma imagem do usuario recupera equipamentos semelhantes antes
da busca textual, permitindo identificar itens que o modelo visual publico nao
conhece.

Etapa 15 (upload pela API): implementada. `POST /documents` e
`POST /documents/images` recebem arquivos por multipart, respondem `202` com um
job acompanhavel e indexam em segundo plano; `GET /documents` e
`DELETE /documents/{id}` completam a gestao do acervo sem acesso ao host.

## Requisitos

- Python 3.11 ou superior

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[knowledge,rag,web,dev]'
python -m pip install 'langchain-openrouter==0.2.8'
cp .env.example .env
```

## Executar

```bash
uvicorn app.main:app --reload
```

A verificacao de liveness fica em `http://127.0.0.1:8000/health`; readiness fica
em `/ready` e metricas em `/metrics`.

Em desenvolvimento, o endpoint de chat recebe `conversationId` e `message`:

```bash
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"conversationId":"demo-1","message":"Como renovar o token?"}'
```

Para enviar uma imagem local junto do prompt:

```bash
IMAGE_BASE64=$(base64 -w0 ./painel.png)
curl -N http://127.0.0.1:8000/chat/stream \
  -H 'Accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  --data-binary @- <<JSON
{
  "conversationId": "imagem-1",
  "tenantId": "public",
  "message": "Identifique o equipamento, os alarmes e o procedimento aplicável.",
  "images": [{
    "name": "painel.png",
    "mimeType": "image/png",
    "data": "$IMAGE_BASE64"
  }]
}
JSON
```

`VISION_MODEL` pode selecionar outro modelo; vazio reutiliza `LLM_MODEL`. O
modelo configurado precisa aceitar imagens. Limites e fluxo estao em
[`docs/multimodal.md`](docs/multimodal.md).

Com o modo SQLite da configuracao de exemplo, reutilizar o mesmo
`conversationId` mantem o contexto inclusive depois de reiniciar a API. Em
producao, envie tambem `tenantId` e `X-API-Key`; o thread passa a ser derivado
do principal, tenant e ID da conversa.

O fallback concreto usa Tavily e fica desativado por padrao. Para permitir
buscas externas, defina `WEB_SEARCH_ENABLED=true` e `TAVILY_API_KEY`. Sem o
fallback, evidencia interna insuficiente termina de forma segura em
`source=none`.

## Executar com Docker Compose

Prepare a configuracao e suba API e Chroma:

```bash
cp .env.example .env
docker compose build api
docker compose up -d api
docker compose ps
curl http://127.0.0.1:8000/ready
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

Para equipamentos proprietarios, cadastre uma ou mais fotos de referencia e
associe os manuais ao mesmo identificador:

```bash
cp foto-frontal.png documents/foto-frontal.png
cp manual-olt.md documents/manual-olt.md

docker compose build index-image ingest api
docker compose run --rm index-image \
  --equipment-id OLT-9000 \
  --equipment-name "OLT proprietaria" \
  --manufacturer CPQD \
  --model X9000 \
  --description "Chassi frontal com oito portas opticas" \
  --tenant-id public \
  /documents/foto-frontal.png

docker compose run --rm ingest \
  --tenant-id public \
  --equipment-id OLT-9000 \
  /documents/manual-olt.md
docker compose up -d --force-recreate api
```

Cadastre fotos de diferentes angulos com o mesmo `equipment_id` para aumentar
a cobertura. A primeira execucao baixa o modelo visual configurado. O limiar
`VISUAL_MAX_DISTANCE` precisa ser calibrado com imagens reais; valores menores
exigem maior similaridade.

## Subir documentos pela API

Alem dos CLIs, a propria API recebe arquivos. O envio responde `202` com um job,
porque a ingestao carrega o modelo de embedding e pode levar minutos:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F 'file=@manual-olt.pdf' \
  -F 'tenantId=public' \
  -F 'equipmentId=OLT-9000'

curl 'http://127.0.0.1:8000/documents/jobs/JOB_ID?tenantId=public'
```

Fotos de referencia do catalogo visual usam `POST /documents/images` com
`equipmentId` e `equipmentName`. `GET /documents` lista o acervo e
`DELETE /documents/{documentId}` remove os chunks e o arquivo enviado. Formatos,
limites, codigos de erro e isolamento por tenant estao em
[`docs/uploads.md`](docs/uploads.md).

Para uma demonstracao completa em curl, com os arquivos de `documents/`:

```bash
scripts/ambiente.sh subir        # constroi a imagem, sobe chroma + api e espera o /ready
scripts/ambiente.sh aquecer      # opcional: carrega os modelos antes da apresentacao
scripts/ambiente.sh demo         # roteiro multimodal: identifica o aparelho pela foto
scripts/ambiente.sh monitoracao  # roteiro de alarmes ao vivo, do zero
```

No roteiro, nenhuma pergunta cita o aparelho: a indicacao vem da foto anexada ao
prompt. Ele carrega a especificacao e a foto do primeiro aparelho e pergunta com
essa foto; repete a pergunta com a foto de outro aparelho e recebe a recusa
segura, porque o manual dele ainda nao foi indexado; carrega esse manual e faz a
mesma pergunta com a mesma foto, agora respondida. `scripts/ambiente.sh` ainda
aceita `status`, `logs`, `parar` e `limpar`. A imagem precisa ser reconstruida
porque os endpoints de upload dependem de `python-multipart` e `pypdf`.

`scripts/ambiente.sh monitoracao` mostra o outro eixo e cuida de tudo sozinho:
sobe a API se preciso, reconstroi a imagem se ela estiver sem os extras, sobe um
servidor MCP de mentira (dados inventados, so para exercitar o caminho numa
maquina sem Zabbix), escreve as chaves `MONITORING_*` no `.env`, recria a API
para que ela as leia, lista as ferramentas publicadas e so entao roda o roteiro.
Nele o classificador decide se a pergunta precisa de dado ao vivo, so entao o
Zabbix e consultado, e uma consulta que volta vazia termina em "nao encontrei"
em vez de escorregar para a web.

Os passos tambem existem soltos: `scripts/ambiente.sh zabbix-falso` sobe e
configura so o servidor falso, `scripts/ambiente.sh ferramentas` lista o que um
servidor MCP publica -- que e como se descobrem os nomes exatos para a allowlist
-- e `scripts/ambiente.sh langfuse` sobe a stack de tracing.

Depois de ingerir os documentos, o retriever pode ser executado isoladamente:

```bash
docker compose run --rm retrieve "Como renovar um access token?"
```

Para executar o RAG completo, configure `LLM_MODEL` e as credenciais exigidas
pelo provider de LLM no `.env`. O fallback Tavily e opcional e requer
`WEB_SEARCH_ENABLED=true` e `TAVILY_API_KEY`. A imagem usa
`langchain-openrouter==0.2.8` por padrao; o build arg
`LLM_PROVIDER_PACKAGE` permite substituir esse adapter:

```bash
docker compose build rag
docker compose run --rm rag "Como renovar um access token?"
```

A base do servidor Chroma, os checkpoints e o cache dos modelos ficam em
volumes nomeados. `docker compose down` os preserva; `docker compose down -v`
tambem os remove e apaga o conhecimento indexado e os threads. Consulte
[`docs/containers.md`](docs/containers.md) para os detalhes.

## Testar

```bash
scripts/ambiente.sh testes              # suite + avaliacao offline + avaliacao visual
scripts/ambiente.sh testes --com-demos  # e mais os dois roteiros interativos
```

Sao quatro camadas independentes: a suite automatizada (`pytest`), a avaliacao
offline da recuperacao contra o baseline versionado, a varredura de limiar da
identificacao visual e os dois roteiros em curl. As duas primeiras rodam em
qualquer maquina, sem Chroma, modelo baixado ou credencial; a terceira exige o
catalogo visual indexado, o que `scripts/ambiente.sh demo` faz.

Para rodar so a suite:

```bash
pytest
```

O plano de entregas e os gates entre iteracoes estao em
[`docs/implementation-roadmap.md`](docs/implementation-roadmap.md).
Os controles e o checklist de deploy estao em
[`docs/production.md`](docs/production.md), o tracing de LLM em [`docs/tracing.md`](docs/tracing.md), a consulta a monitoracao em [`docs/monitoring.md`](docs/monitoring.md).

## Executar a avaliacao

O baseline local nao exige Chroma, modelos baixados ou credenciais:

```bash
agentic-rag-evaluate evaluation/datasets/project-docs-v1.json \
  --offline \
  --name offline-project-docs-v1 \
  --output evaluation/baselines/offline-project-docs-v1.json
```

Para avaliar os modelos e a collection configurados, remova `--offline`. Use
`--retrieval-only` quando nao quiser chamar o LLM e `--baseline CAMINHO` para
obter os deltas contra um relatorio anterior. O contrato das metricas, o fluxo
de ingestao do corpus e os limites do baseline estao em
[`docs/evaluation.md`](docs/evaluation.md).

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
  --chunk-overlap 100 \
  --tenant-id public
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

## Executar o RAG basico

Defina o identificador `provider:model` e a credencial esperada pelo adapter.
Por exemplo, para o OpenRouter:

```bash
export LLM_MODEL=openrouter:anthropic/claude-sonnet-4.6
export OPENROUTER_API_KEY=...
export WEB_SEARCH_ENABLED=true
export TAVILY_API_KEY=...
agentic-rag-ask "Como renovar um access token?"
```

A CLI consulta o Chroma configurado, classifica os chunks e so os passa ao
modelo de resposta quando o grader considera a evidencia suficiente. Caso
contrario, pode reescrever a consulta e tentar novamente dentro do limite
configurado. Se o RAG continuar insuficiente, executa a busca web apenas quando
ela estiver habilitada e avalia os resultados antes de responder. A saida
publica continua contendo `answer` e
`source`, agora com valores `rag`, `web` ou `none`, e passa a incluir `sources`
com a proveniencia citada. Os contratos estao detalhados em
[`docs/basic-rag.md`](docs/basic-rag.md),
[`docs/rag-grader.md`](docs/rag-grader.md) e
[`docs/query-rewrite.md`](docs/query-rewrite.md). O protocolo HTTP/SSE esta em
[`docs/streaming.md`](docs/streaming.md) e a memoria conversacional em
[`docs/memory.md`](docs/memory.md). O fallback externo esta em
[`docs/web-fallback.md`](docs/web-fallback.md) e o contrato de proveniencia em
[`docs/citations.md`](docs/citations.md). O perfil de producao esta em
[`docs/production.md`](docs/production.md) e a consulta por imagem em
[`docs/multimodal.md`](docs/multimodal.md).
