# Plano iterativo de implementacao

Este plano traduz o roadmap arquitetural v1.0 em incrementos verificaveis. Uma
etapa so muda para **aceita** depois que seus criterios forem demonstrados; a
seguinte nao deve ser implementada antes disso.

## Quadro de etapas

| Etapa | Incremento | Evidencia para aceite | Estado |
|---:|---|---|---|
| 1 | Bootstrap FastAPI, settings, `GET /health`, testes e comandos locais | API inicia sem servicos externos; `/health` responde 200; testes passam | Aceita |
| 2 | Ingestao: loader inicial, normalizacao, metadata, chunking, embeddings e Chroma | Documento de teste gera chunks validos; persistencia e reindexacao deterministica comprovadas | Aceita |
| 3 | `KnowledgeRetriever`: dense retrieval, MMR e CrossEncoder | Consultas conhecidas recuperam chunks esperados e o reranker altera a ordem em caso controlado | Implementada, aguardando aceite e calibracao no corpus real |
| 4 | RAG basico: prepare, retrieve e answer no LangGraph | Caminho ponta a ponta grounded nos documentos, com nodes isoladamente testaveis | Implementada; testes isolados passam, aguardando aceite com corpus e LLM configurados |
| 5 | Grader RAG estruturado e rota temporaria de `no_answer` | Casos relevantes e insuficientes separados por saida validada | Implementada; testes dos dois ramos passam, aguardando aceite com corpus e LLM configurados |
| 6 | Rewrite e retry limitado | Numero maximo de tentativas comprovado e ausencia de loop infinito | Implementada; testes comprovam sucesso apos rewrite e termino exato no limite |
| 7 | SSE em `POST /chat/stream` com adapter publico | Status precede tokens; somente resposta final e transmitida; termina em `done` ou `error` | Implementada; contrato HTTP e filtragem de eventos cobertos por testes |
| 8 | Threads e memoria por `conversationId` | Follow-up vira consulta independente e conversas ficam isoladas | Implementada; criterios cobertos por testes com checkpointer real |
| 9 | Fallback web por interface substituivel | Web so executa apos falha interna; evidencia insuficiente resulta em `no_answer` | Implementada; ambos os gates e metadata de URL cobertos por testes |
| 10 | Citacoes e proveniencia normalizadas | Toda citacao resolve para uma fonte real com metadata preservada | Implementada; validacao fechada e contrato SSE cobertos por testes |
| 11 | Dataset e avaliacao reproduzivel | Baseline versionado com Recall@K e MRR ou nDCG, alem de grounding | Implementada; baseline offline e runner live cobertos por testes |
| 12 | Hardening de producao | Persistencia duravel, resiliencia, seguranca, observabilidade e deploy validados | Implementada; controles e falhas principais cobertos por testes |
| 13 | Consulta multimodal por imagem | Imagem vira evidencia visual citavel e consulta para o RAG sem persistencia do binario | Implementada; validacao, analise, retrieval e SSE cobertos por testes |
| 14 | Catalogo visual proprietario | Imagem recupera referencias visuais e restringe manuais pelo `equipment_id` | Implementada; ingestao visual, limiar, tenant, fusao e SSE cobertos por testes |
| 15 | Upload de documentos e imagens por HTTP | Arquivo enviado vira job consultavel e chunks indexados sob o tenant | Implementada; upload, job, listagem e remocao cobertos por testes |
| 16 | Warmup de modelos e prontidao honesta | `/ready` permanece 503 ate embedder, reranker e CLIP estarem carregados | Implementada; aguardando aceite em deploy real |
| 17 | Resolucao de equipamento por multiplos sinais | Identificador lido na imagem prevalece sobre similaridade visual; abstencao no lugar do chute | Implementada; testes cobrem prevalencia, abstencao e ambiguidade |
| 18 | Avaliacao da identificacao visual | Dataset versionado com positivos e negativos, e limiar recomendado por varredura | Implementada; aguardando aceite com fotos reais do catalogo |
| 19 | Observabilidade de LLM com Langfuse self-hosted | Um trace por requisicao cobrindo visao e grafo, e o servico responde normalmente com a stack desligada | Implementada; degradacao e agrupamento cobertos por testes, aguardando aceite com a stack no ar |
| 20 | Consulta a monitoracao ao vivo via MCP | Pergunta sobre alarmes consulta o servidor MCP e a resposta cita a ferramenta que produziu o dado | Implementada; roteamento, allowlist, ponte e degradacao cobertos por testes |
## Iteracao intercalada: containers de desenvolvimento

Apos a Etapa 2, foi solicitada uma base de containerizacao antes do retriever.
Ela inclui targets separados para API, knowledge e RAG, usuario sem
privilegios, servidor Chroma em container, readiness HTTP e volumes para
dados/cache de modelos. Na Etapa 12, a mesma base recebeu persistencia de
checkpoints e reducao adicional de privilegios.

## Ciclo de cada iteracao

1. Confirmar o gate da etapa anterior.
2. Especificar um pequeno conjunto de casos de aceite automatizaveis.
3. Implementar somente o incremento da etapa corrente.
4. Executar testes unitarios e uma verificacao manual proporcional ao risco.
5. Registrar decisoes, resultados, riscos e atualizar este quadro.
6. Solicitar aceite antes de iniciar a etapa seguinte.

## Etapa 4 implementada

Foi integrado apenas o caminho simples `prepare_query -> retrieve_rag ->
generate_answer`. O grafo consome o contrato do `KnowledgeRetriever`, sem
conhecer Chroma, MMR ou CrossEncoder. O prompt limita a resposta aos chunks
recuperados e a ausencia de documentos encerra com uma resposta segura sem
chamar o LLM. Cada node e o grafo compilado possuem testes com dependencias
injetadas.

## Etapa 5 implementada

Foi adicionado `grade_rag` depois do retrieval. O contrato Pydantic exige
`relevant` booleano, `confidence` entre zero e um e `reason` nao vazio. A rota
condicional libera `generate_answer` somente para contexto suficiente; os
demais casos terminam em `no_answer`. Retrieval vazio e tratado sem chamada ao
modelo e saidas estruturadas invalidas interrompem o fluxo em vez de liberar
uma resposta sem decisao valida.

## Etapa 6 implementada

Foi adicionado `rewrite_query` entre um grade negativo e uma nova execucao de
`retrieve_rag`. `retrieval_attempts` conta apenas retrievals e
`MAX_RAG_ATTEMPTS` representa o total, incluindo a consulta original. Um grade
positivo responde imediatamente; um grade negativo reescreve somente enquanto
ha budget e termina em `no_answer` ao atingir o limite. Os testes cobrem
resposta na primeira tentativa, sucesso apos rewrite e esgotamento exato sem
loop infinito.

## Etapa 7 implementada

Foi criado `POST /chat/stream` com request `conversationId`/`message` e um
adapter separado que converte `updates` e `messages` do LangGraph em eventos
SSE publicos. O stream inicia com `status`, emite `sources` antes dos tokens,
filtra mensagens para aceitar apenas `generate_answer` e termina com `done` ou
um `error` sem detalhes internos. Na Etapa 10, `sources` passou a carregar apenas
fontes citadas e validadas. Na Etapa 8, `conversationId` passou a ser usado como
thread.

## Etapa 8 implementada

O adapter mapeia a identidade da conversa para `configurable.thread_id`. Em
desenvolvimento o grafo pode usar `InMemorySaver`; a API e o Compose usam o
checkpointer configurado. O estado acumula pares de mensagens apenas ao
terminar uma resposta. `prepare_query` usa esse
historico para resolver referencias e produz `prepared_query`; retrieval,
grader, rewrite e geracao recebem a consulta autocontida, nunca o historico
bruto. Testes comprovam que "E no Keycloak?" herda o assunto OAuth apenas no
mesmo thread e que outro ID inicia sem contexto.

Na Etapa 12, o modo SQLite passou a preservar threads entre reinicios em uma
instancia. O thread da API e derivado do principal, tenant e `conversationId`,
sem persistir o ID publico diretamente.

## Etapa 9 implementada

Foi criado `WebSearchProvider`, com adapter Tavily configuravel e sem referencia
ao provider dentro dos nodes. Um grade RAG negativo no limite segue para
`search_web -> grade_web`; apenas `web_relevant=true` libera o gerador com
`source=web`. Resultado vazio ou insuficiente termina em `no_answer`. Os
resultados preservam titulo, URL, score e provider. A Etapa 10 acrescentou o
contrato publico de citacoes sobre essa metadata.

Testes comprovam que o provider nao e chamado quando o RAG e suficiente, que a
ultima consulta interna alimenta o fallback, que URLs permanecem na metadata e
que evidencia web reprovada nunca chega ao gerador.

## Etapa 10 implementada

Foi criado um modelo `Source` unico para RAG e web. Fontes internas preservam o
`chunkId` e metadata de titulo, pagina, secao e URL quando disponivel; fontes
externas preservam a URL original. A evidencia recebe IDs deterministas antes
da geracao, e o prompt permite apenas os rotulos fornecidos.

A resposta terminal e validada duas vezes: o node rejeita ausencia de citacao
ou IDs desconhecidos, e o adapter SSE confirma que resposta, tipo agregado e
lista publica sao consistentes. Apenas depois dessa validacao o evento
`sources` e os tokens sao enviados. Testes comprovam que `[2]` sem fonte
correspondente termina em `error` sem vazar texto.

## Etapa 11 implementada

Foi versionado um dataset inicial com perguntas sobre contratos reais do
projeto e fontes esperadas. O runner calcula Recall@K, MRR, latencias e metricas
deterministicas de relevancia lexical, grounding por chunks citados, validade
das citacoes e cobertura das fontes esperadas. Dataset e corpus possuem hashes
independentes, impedindo comparacoes entre revisoes diferentes.

Cada relatorio registra chunking, embedding, MMR, reranker, gerador, tokens e
precos fornecidos. O comparador apresenta deltas de qualidade, latencia e custo
e destaca configuracoes alteradas. Um baseline offline, explicitamente separado
dos providers de producao, permite reproduzir o gate sem Chroma ou credenciais;
o mesmo comando avalia a collection e os modelos reais quando configurados.

## Etapa 12 implementada

O runtime passou a possuir um checkpointer SQLite lazy com WAL e volume
dedicado, fechado no shutdown da aplicacao. A configuracao de producao exige
autenticacao, politicas de tenant, rate limit, logs JSON, persistencia e
protecao de metricas. API keys sao mantidas como secrets e comparadas por
digest; o thread e derivado da identidade e o retrieval filtra a metadata
`tenant_id`.

Chamadas de LLM receberam timeout e retries limitados. A busca web acrescenta
backoff exponencial e circuit breaker para falhas transientes. Streams liberam
recursos ao desconectar, respeitam deadline total e retornam erros publicos sem
detalhes internos. `/ready`, `/metrics`, `X-Request-ID` e metricas de HTTP,
streams, autenticacao e rate limit formam a base do SLO operacional.

O container da API usa filesystem raiz somente leitura, usuario sem privilegio,
capabilities removidas, init, shutdown gracioso e volume de checkpoints. SQLite
e o rate limiter em memoria limitam esta entrega a uma replica; escala
horizontal requer checkpointer e quotas compartilhados. O runbook completo
esta em [`production.md`](production.md).

## Etapa 13 implementada

`POST /chat/stream` aceita imagens locais em Base64 com MIME e nome. Antes de
abrir o grafo persistente, um modelo com visao produz uma estrutura validada com
resumo, observacoes, OCR, identificadores, incertezas e consulta de retrieval.
O binario e descartado nesse limite; somente a analise textual entra no estado
e no historico.

A analise visual se torna um `RetrievedDocument` transitorio com identificador
derivado do hash da imagem. Assim, pode participar do grade e receber citacao,
enquanto a consulta enriquecida recupera manuais e procedimentos no Chroma. A
entrega cobre imagem como consulta. O catalogo visual permanente e o vinculo
com manuais foram acrescentados na Etapa 14.

## Etapa 14 implementada

Foi criada a collection Chroma `equipment_images`, separada da collection
textual porque o espaco vetorial CLIP tem dimensao e semantica proprias. A CLI
`agentic-rag-index-image` transforma cada foto de referencia em embedding e
grava `equipment_id`, nome, fabricante, modelo, descricao, tenant, hash e
origem. Varias imagens podem compartilhar o mesmo equipamento.

Durante o chat, os bytes recebidos sao comparados com esse catalogo antes de
serem descartados. Os candidatos dentro de `VISUAL_MAX_DISTANCE` entram como
evidencias citaveis e seus IDs formam um filtro `$in` para a busca de manuais,
combinado ao filtro de tenant. O modelo recebe candidatos e distancias como
evidencia, sem transformar similaridade visual em certeza automatica.

## Etapa 15 implementada

A alimentacao do acervo deixou de exigir acesso ao host. `POST /documents` e
`POST /documents/images` recebem arquivos por multipart, gravam em `UPLOAD_DIR`
com nome saneado e caminho deterministico por tenant, e enfileiram a ingestao em
um worker proprio; a resposta e `202` com um job consultavel em
`GET /documents/jobs/{id}`. O caminho deterministico faz com que reenviar o
mesmo nome substitua a versao anterior em vez de duplicar chunks.

O pipeline textual passou a aceitar PDF pela camada de texto, com cada pagina
marcada como um cabecalho para que o numero chegue as citacoes.
`GET /documents` agrega os chunks por documento e `DELETE /documents/{id}`
remove indice e arquivo, sempre sob o mesmo filtro de tenant do chat. Detalhes
em [`uploads.md`](uploads.md).

## Etapa 16 implementada

O runtime ganhou `warmup()`, que carrega grafo, analisador de imagem e provider
visual antes do primeiro request. A aplicacao dispara isso em uma thread durante
o startup quando `WARMUP_ON_STARTUP` esta ativo, e a prontidao passou a exigir
os modelos em memoria, nao apenas a ausencia de erro. Carregar fora do event
loop mantem `/ready` respondendo 503 durante o processo, em vez de deixar a
replica sem aceitar conexao e indistinguivel de uma que caiu.

## Etapa 17 implementada

`EquipmentResolver` decide qual equipamento a foto mostra fundindo o
identificador lido por OCR com a similaridade visual, e devolve confianca com
direito a abstencao. A distancia do Chroma vira cosseno, o casamento textual
normaliza caixa, acentos e pontuacao, e valores curtos demais nao casam por
substring.

Os candidatos permanecem evidencia citavel em qualquer desfecho; apenas um
equipamento efetivamente resolvido entra no filtro `$in` dos manuais. Essa
separacao troca o modo de falha: antes um falso positivo escondia o manual
correto, agora a busca segue sem filtro. O resolvedor nao importa LangChain, o
que o mantem utilizavel pela imagem `knowledge` e pela avaliacao.

## Etapa 18 implementada

`agentic-rag-evaluate-visual` mede a identificacao com um dataset versionado e
recomenda o limiar. Cada imagem e consultada uma vez com o portao aberto e a
varredura reaplica os demais limiares sobre as distancias obtidas. Os desfechos
separam acerto, erro, abstencao indevida e abstencao correta em negativos, e o
limiar recomendado maximiza `accuracy - false_positive_rate`, tratando uma
identificacao errada como tao cara quanto uma certa e util.

## Etapa 19 implementada

`app/observability/tracing.py` adiciona tracing opcional por Langfuse
self-hosted. Cada requisicao abre uma observacao raiz e `propagate_attributes`
mantem a analise visual e a execucao do grafo no mesmo trace, com o thread
derivado como `session_id` -- nunca o `conversationId` publico.

O modulo degrada para no-op em toda falha: pacote ausente, credencial faltando,
cliente que nao constroi ou span que nao abre. Apenas a entrada do contexto e
protegida, entao erros do proprio request continuam propagando. O cliente recebe
host e chaves explicitamente das configuracoes, sem depender de qual variavel de
ambiente a versao instalada do SDK honra.

A stack (web, worker, Postgres, ClickHouse, Redis e MinIO) fica atras do profile
`observability` do Compose, com os servicos prefixados para nao colidirem com os
do projeto. Detalhes em [`tracing.md`](tracing.md).

## Etapa 20 implementada

`app/monitoring/` faz o agente consultar um servidor MCP de monitoracao quando a
pergunta pede dado ao vivo. Um classificador decide antes do `prepare_query`, de
modo que o alarme recuperado passa a fazer parte da consulta que busca os
manuais; se rodasse depois, a busca textual ja teria acontecido com a pergunta
vaga.

Cada chamada de ferramenta vira `RetrievedDocument` com locator
`zabbix:<ferramenta>` e os argumentos que a produziram, preservando o contrato de
citacao. A fronteira de seguranca e `MONITORING_ALLOWED_TOOLS`: so as ferramentas
nomeadas chegam ao modelo, entao um servidor que publica a API de escrita nao e
alcancavel a partir de um chat.

O pipeline e sincrono e o cliente MCP e async-only, entao `bridge.py` mantem um
event loop proprio numa thread. Falha de ferramenta vira observacao de erro em vez
de excecao, e monitoracao indisponivel degrada para documentacao apenas. Detalhes
em [`monitoring.md`](monitoring.md).

## Etapa 21 implementada

Uma consulta ao Zabbix passa a terminar de tres formas distintas em vez de duas.
Vazio e um fato -- `problem_get` devolvendo `[]` significa que nao ha alarmes --
e continua virando evidencia citavel, com o corpo escrito em palavras para o
gerador nao ler `[]` como ausencia de contexto. Ja "nao consegui consultar" e a
falta do fato.

As duas cortam o fallback: uma pergunta que foi a monitoracao e voltou sem dado
nao reescreve a consulta nem cai na web, porque nenhum manual e nenhuma pagina da
internet sabe o estado atual da infraestrutura do cliente. O adaptador de stream
acompanha a decisao, para nao anunciar uma reformulacao que nao vai acontecer.

Os scripts da POC cobrem o caminho: `scripts/demo-monitoracao.sh` demonstra os
quatro casos e `scripts/zabbix-falso.py` (profile `demo`) permite rodar isso sem
um Zabbix. Nesta etapa tambem se corrigiu o `Dockerfile`, que construia os wheels
de `tracing` e `monitoring` mas instalava so `[knowledge,rag,web]`: as duas
features estavam ausentes da imagem.

Com esta entrega, as etapas 1 a 21 foram implementadas. As proximas iteracoes
devem ser dirigidas pelos resultados do corpus real, SLOs de producao e
requisitos de escala.
