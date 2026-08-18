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
| 4 | RAG basico: prepare, retrieve e answer no LangGraph | Caminho ponta a ponta grounded nos documentos, com nodes isoladamente testaveis | Pendente; ha apenas estado preexistente nao validado |
| 5 | Grader RAG estruturado e rota temporaria de `no_answer` | Casos relevantes e insuficientes separados por saida validada | Pendente |
| 6 | Rewrite e retry limitado | Numero maximo de tentativas comprovado e ausencia de loop infinito | Pendente |
| 7 | SSE em `POST /chat/stream` com adapter publico | Status precede tokens; somente resposta final e transmitida; termina em `done` ou `error` | Pendente |
| 8 | Threads e memoria por `conversationId` | Follow-up vira consulta independente e conversas ficam isoladas | Pendente |
| 9 | Fallback web por interface substituivel | Web so executa apos falha interna; evidencia insuficiente resulta em `no_answer` | Pendente |
| 10 | Citacoes e proveniencia normalizadas | Toda citacao resolve para uma fonte real com metadata preservada | Pendente |
| 11 | Dataset e avaliacao reproduzivel | Baseline versionado com Recall@K e MRR ou nDCG, alem de grounding | Pendente |
| 12 | Hardening de producao | Persistencia duravel, resiliencia, seguranca, observabilidade e deploy validados | Pendente |

## Iteracao intercalada: containers de desenvolvimento

Apos a Etapa 2, foi solicitada uma base de containerizacao antes do retriever.
Ela inclui targets separados para API e ingestao, usuario sem privilegios,
servidor Chroma em container, readiness HTTP e volumes para dados/cache de
modelos. Esta entrega facilita o desenvolvimento reproduzivel, mas nao antecipa
os controles de producao da Etapa 12.

## Ciclo de cada iteracao

1. Confirmar o gate da etapa anterior.
2. Especificar um pequeno conjunto de casos de aceite automatizaveis.
3. Implementar somente o incremento da etapa corrente.
4. Executar testes unitarios e uma verificacao manual proporcional ao risco.
5. Registrar decisoes, resultados, riscos e atualizar este quadro.
6. Solicitar aceite antes de iniciar a etapa seguinte.

## Proxima iteracao proposta: Etapa 4

Integrar apenas o caminho simples no LangGraph: `prepare_query`, `retrieve_rag`
e `generate_answer`. O grafo deve consumir `KnowledgeRetriever` sem conhecer
Chroma, MMR ou CrossEncoder. Grader, rewrite e fallback web permanecem fora
dessa iteracao.
