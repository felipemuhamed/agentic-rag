# RAG basico no LangGraph

A Etapa 4 conectou o retriever da Etapa 3 a um gerador de resposta por um grafo
linear e deliberadamente pequeno. Na Etapa 5, o grader foi inserido entre o
retrieval e a geracao; na Etapa 6, um ciclo limitado passou a reescrever
consultas insuficientes:

```text
START -> prepare_query -> retrieve_rag -> grade_rag
                                          | relevant=true  -> generate_answer -> END
                                          | false e ha budget -> rewrite_query
                                          |                      -> retrieve_rag
                                          | false no limite  -> no_answer -> END
```

## Estado e contratos

A entrada do grafo contem somente `question`; a saida contem `answer` e
`source`, cujo valor e `rag` ou `none`. O estado interno adiciona `messages`,
`prepared_query`, `search_query`, `retrieval_attempts`, documentos recuperados,
diagnosticos do retrieval e a decisao do grader (`rag_relevant`,
`rag_confidence` e `rag_reason`). Nao ha estado de web.

Os nodes recebem dependencias por interface:

- um retriever que retorna `RetrievalResult`;
- um grader que retorna `RagGrade` validado;
- um query preparer que resolve referencias usando o historico do thread;
- um query rewriter que produz uma nova consulta textual;
- um gerador que recebe a pergunta e a lista de `RetrievedDocument`.

Assim, o grafo nao conhece detalhes de Chroma, embeddings, MMR, CrossEncoder ou
provider de chat e pode ser exercitado inteiramente com doubles de teste.

## Grounding

`generate_answer` envia ao modelo apenas a consulta preparada e os chunks
finais do retrieval, serializados com `chunk_id`, conteúdo e metadata. A instrucao de
sistema proibe conhecimento externo, suposicoes e obediencia a instrucoes
presentes dentro dos chunks.

Quando nenhum documento e recuperado, o grader produz uma decisao negativa sem
chamar seu modelo. A rota pode então reescrever a consulta se ainda houver
budget. Com ou sem documentos, apenas uma decisao estruturada `relevant=true`
libera `generate_answer`. Consulte [`rag-grader.md`](rag-grader.md) para o
contrato da decisao e [`query-rewrite.md`](query-rewrite.md) para o ciclo.

## Execucao

Instale o extra do grafo e o adapter do provider escolhido. Exemplo com OpenAI:

```bash
python -m pip install -e '.[knowledge,rag]'
python -m pip install 'langchain-openai>=1.5,<2'
export LLM_MODEL=openai:gpt-4.1-mini
export OPENAI_API_KEY=...
agentic-rag-ask "Como renovar um access token?"
```

O mesmo fluxo no Compose usa o target `rag`:

```bash
docker compose build rag
docker compose run --rm rag "Como renovar um access token?"
```

O primeiro uso pode baixar embedding e reranker para o `model-cache`. O Chroma
deve estar acessivel e a collection deve ter sido populada pela ingestao.

## Memoria conversacional

Desde a Etapa 8, o grafo pode ser compilado com checkpointer. O endpoint usa
`conversationId` como `thread_id`, e `prepare_query` consulta as mensagens
anteriores somente para converter uma pergunta dependente em
`prepared_query`. O historico bruto nao e enviado ao retriever, grader,
rewriter nem gerador. Consulte [`memory.md`](memory.md).

## Limites atuais

O fluxo ainda nao possui fallback web, persistencia conversacional duravel ou
citacoes normalizadas. O streaming da Etapa 7 fica em um adapter externo ao
grafo.
