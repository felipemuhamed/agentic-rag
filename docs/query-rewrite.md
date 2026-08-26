# Query rewrite e retry limitado

A Etapa 6 cria um ciclo controlado quando o grader considera insuficientes os
documentos recuperados:

```text
retrieve_rag -> grade_rag -> rewrite_query -> retrieve_rag
                    |
                    +-> generate_answer, se relevante
                    +-> search_web, se o limite foi atingido
```

## Semantica do contador

`retrieval_attempts` começa em zero no `prepare_query` e e incrementado somente
por `retrieve_rag`. `MAX_RAG_ATTEMPTS` e o numero total permitido de retrievals,
incluindo a consulta original.

Com o padrao `MAX_RAG_ATTEMPTS=2`, o fluxo permite:

1. retrieval da pergunta original;
2. no maximo um rewrite e um segundo retrieval;
3. resposta, se o segundo grade for positivo; caso contrario, fallback web
   quando habilitado ou `no_answer` quando desabilitado.

O construtor do grafo e as settings rejeitam valores menores que um. Como a
decisao de continuar depende do contador, uma query reescrita igual a anterior
tambem termina no limite e nao cria loop infinito.

## Contrato do rewriter

O rewriter recebe:

- a consulta preparada, que representa a intencao autocontida do turno;
- a consulta usada no ultimo retrieval;
- o motivo estruturado produzido pelo grader.

O prompt pede uma unica consulta autocontida, sem responder à pergunta nem
introduzir fatos, entidades ou restricoes ausentes dos dados fornecidos. Uma
resposta vazia ou em formato nao textual e rejeitada como erro do modelo.

## Estado entre tentativas

Cada rewrite substitui apenas `search_query`. A mensagem original permanece em
`question`; a intencao autocontida permanece em `prepared_query` e e usada pelo
grader, rewriter e gerador. Os documentos, diagnosticos e grade da tentativa
anterior sao sobrescritos na execucao seguinte; historico detalhado das
tentativas pertence a uma etapa posterior de observabilidade e avaliacao.

## Configuracao

```dotenv
MAX_RAG_ATTEMPTS=2
```

O Compose repassa essa variavel aos servicos `api` e `rag`. Grader, rewriter e
gerador usam o mesmo chat model com tags distintas.

## Limites

O ciclo interno continua limitado e independente da busca externa. Ao esgotar
as tentativas, a ultima `search_query` e entregue uma unica vez ao
`WebSearchProvider`. Nao existe retorno da web para o ciclo de rewrite. Citacoes
continuam fora desta etapa.
