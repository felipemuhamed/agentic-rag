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

Com esta entrega, as 14 etapas foram implementadas. As proximas iteracoes devem
ser dirigidas pelos resultados do corpus real, SLOs de producao e requisitos de
escala.
