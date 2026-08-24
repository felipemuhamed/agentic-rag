# Fallback web com gate de evidencia

A Etapa 9 acrescenta conhecimento externo somente depois de o fluxo interno
esgotar `MAX_RAG_ATTEMPTS`.

## Roteamento

```text
grade_rag relevant=true  -> generate_answer(source=rag)
grade_rag false + budget -> rewrite_query -> retrieve_rag
grade_rag false no limite -> search_web -> grade_web
                                           | relevant=true
                                           | -> generate_answer(source=web)
                                           | relevant=false
                                           | -> no_answer(source=none)
```

Nao ha escolha livre de ferramentas pelo modelo. A conditional edge do grafo e
a unica responsavel por acionar web, e existe no maximo uma busca externa por
turno. A busca recebe a ultima `search_query`, que pode ter sido refinada pelo
ciclo interno; o web grader avalia os resultados contra `prepared_query`.

## Limite de provider

`WebSearchProvider.search(query)` e o contrato consumido pelo node. O adapter
concreto atual usa a [Search API da Tavily](https://docs.tavily.com/documentation/api-reference/endpoint/search)
por HTTP. Tavily nao aparece no state, nos nodes, no adapter SSE nem no gerador.

Cada `WebSearchResult` valido preserva:

- `title`;
- URL HTTP(S) absoluta;
- trecho de `content`;
- `score` entre zero e um;
- nome do `provider`;
- metadata com `source`, `url`, `title` e `provider`.

Entradas malformadas do resultado sao descartadas. Uma resposta estruturalmente
invalida do endpoint e tratada como falha, nao como evidencia vazia. A chave de
API nao e validada na construcao do grafo: ela passa a ser obrigatoria somente
se `search_web` for realmente executado.

## Segundo gate

`grade_web` usa structured output validado por Pydantic:

```python
class WebGrade(BaseModel):
    relevant: bool
    confidence: float  # 0.0 <= confidence <= 1.0
    reason: str        # nao vazio
```

URL existente, score alto ou afinidade tematica nao liberam resposta por si
so. O prompt exige conteudo suficiente para responder sem completar lacunas.
Lista vazia produz uma decisao negativa deterministica sem chamar o LLM. Saida
estruturada invalida interrompe o fluxo e vira `error` seguro no SSE.

## Configuracao

```dotenv
WEB_SEARCH_PROVIDER=tavily
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_DEPTH=basic
WEB_SEARCH_TIMEOUT_SECONDS=10
TAVILY_API_KEY=...
TAVILY_BASE_URL=https://api.tavily.com
```

`WEB_SEARCH_MAX_RESULTS` aceita de 1 a 20. As profundidades aceitas sao
`basic`, `advanced`, `fast` e `ultra-fast`. O request desabilita respostas
geradas pelo provider e conteudo bruto; apenas resultados com fonte sao usados
como evidencia pelo nosso proprio grader.

## Execucao e teste

Instalacao local:

```bash
python -m pip install -e '.[knowledge,rag,web,dev]'
```

Depois de configurar Chroma, LLM e Tavily:

```bash
agentic-rag-ask "Pergunta fora da base interna"
```

Os testes nao acessam a internet. Doubles comprovam o roteamento, o gate e o
request do adapter:

```bash
pytest tests/websearch tests/graph/test_web_grader.py tests/graph/test_graph.py
```

## Integracao com citacoes

Desde a Etapa 10, resultados web aprovados sao convertidos em fontes publicas.
Cada citacao mapeia diretamente para a URL original, e somente URLs citadas sao
enviadas no evento SSE `sources`. Consulte [`citations.md`](citations.md).

Retry HTTP, circuit breaker, allowlist de dominios e observabilidade operacional
continuam reservados ao hardening de producao.
