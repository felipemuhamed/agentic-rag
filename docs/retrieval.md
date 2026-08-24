# Knowledge retrieval

A Etapa 3 implementa retrieval como componente independente. Nenhum modulo
desta camada importa LangGraph, API HTTP ou LLM de resposta.

## Fluxo

```text
query
  -> embedding de query
  -> Chroma dense search (fetch_k)
  -> MMR (candidate_k)
  -> CrossEncoder (um score por par query/chunk)
  -> ordenacao e top final_k
  -> RetrievalResult
```

## Contrato

```python
result = retriever.retrieve(
    "Como renovar um access token?",
    filters={"section": "OAuth"},
)
```

`RetrievalResult.documents` contem `chunk_id`, `content`, `metadata`,
`retrieval_score` e `rerank_score`. `diagnostics` registra quantos candidatos
foram buscados, selecionados pelo MMR, reranqueados e retornados.

O `retrieval_score` atual e o negativo da distancia retornada pelo Chroma. Ele
preserva a ordenacao (maior e melhor), mas nao e probabilidade. O
`rerank_score` preserva a saida bruta do modelo; alguns CrossEncoders retornam
logits positivos e negativos. Thresholds so devem ser definidos depois de
calibracao no corpus real.

## Configuracao

| Variavel | Padrao | Funcao |
|---|---:|---|
| `RAG_FETCH_K` | 60 | Candidatos densos solicitados ao Chroma |
| `RAG_CANDIDATE_K` | 20 | Candidatos preservados pelo MMR |
| `RAG_FINAL_K` | 5 | Chunks finais depois do reranking |
| `RAG_LAMBDA_MULT` | 0.65 | Balanco MMR entre relevancia e diversidade |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding de documentos e queries |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | CrossEncoder de reranking |

O contrato exige `fetch_k >= candidate_k >= final_k > 0` e
`0 <= lambda_mult <= 1`.

## Execucao manual

Localmente:

```bash
agentic-rag-retrieve "pergunta conhecida"
```

Com Compose:

```bash
docker compose run --rm retrieve "pergunta conhecida"
```

Na primeira execucao, embedding e reranker sao baixados para o cache de modelos.
Para avaliar qualidade, use perguntas cujo documento esperado seja conhecido e
confira `chunk_id`, `source`, `section`, scores e posicao final. A Etapa 11
formaliza esse processo com dataset, Recall@K, MRR e relatorios comparaveis em
[`evaluation.md`](evaluation.md).
