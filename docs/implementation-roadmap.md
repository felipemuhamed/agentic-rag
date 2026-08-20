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
| 9 | Fallback web por interface substituivel | Web so executa apos falha interna; evidencia insuficiente resulta em `no_answer` | Pendente |
| 10 | Citacoes e proveniencia normalizadas | Toda citacao resolve para uma fonte real com metadata preservada | Pendente |
| 11 | Dataset e avaliacao reproduzivel | Baseline versionado com Recall@K e MRR ou nDCG, alem de grounding | Pendente |
| 12 | Hardening de producao | Persistencia duravel, resiliencia, seguranca, observabilidade e deploy validados | Pendente |

## Iteracao intercalada: containers de desenvolvimento

Apos a Etapa 2, foi solicitada uma base de containerizacao antes do retriever.
Ela inclui targets separados para API, knowledge e RAG, usuario sem
privilegios, servidor Chroma em container, readiness HTTP e volumes para
dados/cache de modelos. Esta entrega facilita o desenvolvimento reproduzivel,
mas nao antecipa os controles de producao da Etapa 12.

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
um `error` sem detalhes internos. Ate a Etapa 10, `sources` e uma lista vazia.
Na Etapa 8, `conversationId` passou a ser usado como thread.

## Etapa 8 implementada

O adapter mapeia `conversationId` diretamente para `configurable.thread_id` e o
grafo de desenvolvimento usa `InMemorySaver`. O estado persistido acumula pares
de mensagens apenas ao terminar uma resposta. `prepare_query` usa esse
historico para resolver referencias e produz `prepared_query`; retrieval,
grader, rewrite e geracao recebem a consulta autocontida, nunca o historico
bruto. Testes comprovam que "E no Keycloak?" herda o assunto OAuth apenas no
mesmo thread e que outro ID inicia sem contexto.

Esse checkpointer existe somente no processo da API: reinicio e replicas nao
compartilham memoria. Persistencia duravel continua reservada ao hardening.

## Proxima iteracao proposta: Etapa 9

Adicionar fallback web atras de uma interface substituivel e acionado somente
depois do esgotamento do RAG interno. A evidencia web tambem devera passar por
um gate antes de chegar ao gerador; citacoes permanecem para a Etapa 10.
