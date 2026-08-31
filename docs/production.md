# Operacao e hardening de producao

A Etapa 12 fecha a primeira versao do roadmap com persistencia conversacional,
controles de acesso, limites de uso, resiliencia e sinais operacionais. Esses
controles protegem a API; a publicacao do Chroma e do endpoint HTTP continua
dependendo da rede e do gerenciador de segredos da plataforma de deploy.

## Perfil de producao

Use `APP_ENVIRONMENT=production`. Nesse perfil, a aplicacao recusa a
inicializacao se qualquer requisito obrigatorio estiver ausente:

- checkpointer duravel;
- autenticacao por API key com politicas de tenant;
- rate limit e limite de streams concorrentes;
- logs estruturados em JSON;
- chave separada para `/metrics`, quando as metricas estiverem habilitadas.

Um exemplo minimo, com valores deliberadamente ficticios, e:

```dotenv
APP_ENVIRONMENT=production
JSON_LOGS=true
CHECKPOINTER_MODE=sqlite
AUTH_ENABLED=true
API_KEY_POLICIES=[{"key":"replace-with-a-random-key-32-chars","principal":"frontend","tenants":["public"]}]
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
MAX_CONCURRENT_STREAMS=4
METRICS_ENABLED=true
METRICS_API_KEY=replace-with-an-independent-metrics-key
```

Nao grave valores reais no repositorio, na imagem ou na linha de comando. Em
producao, injete-os pelo mecanismo de secrets da plataforma e limite quem pode
ler o ambiente do container. A aplicacao nao registra headers de autenticacao,
corpo das mensagens, prompts nem texto de excecoes.

## Autenticacao, tenants e threads

`API_KEY_POLICIES` e uma lista JSON. Cada item precisa ter exatamente `key`,
`principal` e `tenants`; a chave deve possuir ao menos 16 caracteres. Duas
chaves podem coexistir com o mesmo principal durante rotacao sem perder os
threads. `*` pode ser usado em `tenants` para uma identidade administrativa,
mas uma requisicao autenticada ainda precisa declarar `tenantId`.

```http
POST /chat/stream
X-API-Key: <chave do frontend>
Content-Type: application/json

{
  "conversationId": "ticket-42",
  "tenantId": "public",
  "message": "Como renovar o token?"
}
```

O principal e o tenant autorizados entram no isolamento do thread. O valor
gravado pelo checkpointer e um SHA-256 de `principal + tenant + conversationId`,
portanto o ID fornecido pelo cliente nao e persistido diretamente e IDs iguais
de tenants diferentes nao colidem.

O mesmo tenant precisa existir na metadata dos documentos. Para uma collection
nova:

```bash
agentic-rag-ingest documents/guia.md --tenant-id public
```

Ou defina `KNOWLEDGE_TENANT_ID=public`. Quando a autenticacao esta ativa, o
retrieval sempre aplica `tenant_id` como filtro no Chroma. Documentos antigos
sem essa metadata nao aparecem em consultas com tenant e precisam ser
reingeridos. O Chroma do Compose nao possui autenticacao propria e deve ficar
em rede privada.

## Persistencia conversacional

`CHECKPOINTER_MODE=memory` e util somente em testes ou desenvolvimento. O modo
`sqlite` cria o arquivo indicado por `CHECKPOINTER_PATH`, ativa WAL, configura
espera curta para locks e usa serializacao estrita do LangGraph. No Compose, o
arquivo fica em `/data/checkpoints/checkpoints.sqlite`, no volume nomeado
`checkpoint-data`; recriar a API preserva os threads.

Esse desenho suporta uma instancia da API. SQLite nao e o backend indicado
para replicas concorrentes em hosts diferentes. Antes de escalar
horizontalmente, substitua `CheckpointerHandle` por um saver compartilhado,
como PostgreSQL, mantendo o mesmo `thread_id` derivado da identidade. Defina
backup, retencao e exclusao dos checkpoints conforme a politica de dados do
produto.

## Resiliencia e cancelamento

Chamadas ao LLM usam `LLM_TIMEOUT_SECONDS` e no maximo `LLM_MAX_ATTEMPTS`.
Para OpenRouter, o runtime converte os segundos configurados para os
milissegundos esperados pelo SDK. Retries internos do provider ficam
desabilitados no grafo; somente a politica limitada dos nodes executa novas
tentativas, evitando backoffs multiplicados.
Quando `WEB_SEARCH_ENABLED=true`, `TAVILY_API_KEY` e obrigatoria ja na carga da
configuracao. Com a integracao desativada, falta de evidencia interna termina
em `source=none` sem chamada externa. Falhas transientes da busca web ativa
usam retry exponencial, limitado por
`EXTERNAL_MAX_ATTEMPTS`, e circuit breaker configurado por
`CIRCUIT_FAILURE_THRESHOLD` e `CIRCUIT_RESET_SECONDS`. Erros permanentes nao
sao repetidos.

Cada stream possui deadline total em `STREAM_MAX_DURATION_SECONDS`. Ao exceder
o limite, o cliente recebe `stream_timeout`; nas demais falhas recebe apenas
`stream_failed`. Desconexao do cliente cancela a iteracao e libera sua vaga de
concorrencia. Uma chamada bloqueada no SDK so pode observar o cancelamento ao
retornar, por isso os timeouts dos providers devem ser sempre menores que o
deadline do stream.

O rate limiter atual e local ao processo e limita tanto requisicoes por janela
quanto streams simultaneos por principal e tenant. Em multiplas replicas, use
um limitador compartilhado no gateway ou em um datastore central.

## Health, readiness, logs e metricas

- `GET /health` comprova apenas que o processo HTTP esta vivo e nao inicializa
  modelos nem servicos externos;
- `GET /ready` retorna `503` depois de uma falha na inicializacao lazy do
  runtime e e o endpoint usado pelo healthcheck do container;
- toda resposta inclui `X-Request-ID`; um ID valido fornecido pelo gateway e
  preservado e os demais sao substituidos;
- logs JSON incluem request ID, rota, status, duracao e resultado do stream;
- `GET /metrics` usa o formato Prometheus e exige `X-Metrics-Key` quando uma
  chave foi configurada.

As metricas expostas possuem labels limitados:

- `agentic_rag_http_requests_total` e duracao HTTP;
- `agentic_rag_streams_total`, duracao e streams ativos;
- `agentic_rag_auth_failures_total`;
- `agentic_rag_rate_limit_rejections_total`.

Alertas iniciais podem observar taxa de `5xx`, streams com `error`/`timeout`,
latencia p95, saturacao de streams e rejeicoes do rate limiter. A Etapa 11
continua sendo o gate offline de qualidade; essas metricas medem o SLO online.

## Warmup e prontidao

Com `WARMUP_ON_STARTUP=true`, a aplicacao carrega o modelo de embedding, o
reranker e o CLIP em uma thread propria durante o startup, e `/ready` responde
503 ate que os tres estejam em memoria. Sem isso a prontidao e otimista: um
container recem-subido se declara pronto antes de qualquer modelo existir, o
balanceador manda trafego real para ele e a primeira requisicao gasta o
`STREAM_MAX_DURATION_SECONDS` inteiro carregando pesos em vez de responder.

O carregamento roda fora do event loop de proposito. Bloquear o startup faria o
servidor nao aceitar conexao nenhuma, e a replica ficaria indistinguivel de uma
que caiu; respondendo 503 ela informa o que esta acontecendo. Se o warmup
falhar, a prontidao permanece fechada e o motivo vai para o log -- que e
exatamente o sinal que o orquestrador precisa para nao rotear trafego.

## Container e deploy

O servico `api` do Compose executa como usuario sem privilegios, remove todas
as capabilities, bloqueia ganho de privilegio, usa filesystem raiz somente
leitura, `/tmp` temporario, volume dedicado ao checkpointer, init e shutdown
gracioso. API e Chroma sao publicados apenas em `127.0.0.1` por padrao.

Valide antes do deploy:

```bash
docker compose config --quiet
docker compose build api
docker compose up -d api
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail -H 'X-Metrics-Key: <chave>' http://127.0.0.1:8000/metrics
pytest
```

TLS, WAF, limite distribuido, rotacao de secrets e autenticacao do Chroma
devem ser fornecidos pelo ingress/gateway e pela plataforma. Nao publique a
configuracao de Compose diretamente na Internet.
