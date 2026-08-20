# Streaming SSE para o frontend

A Etapa 7 expoe um protocolo publico estavel e mantem o formato nativo de
streaming do LangGraph dentro de `app/streaming`.

## Endpoint

```http
POST /chat/stream
Content-Type: application/json
Accept: text/event-stream

{
  "conversationId": "demo-1",
  "message": "Como renovar o token?"
}
```

Os dois campos sao obrigatorios, textuais e nao podem conter apenas espacos.
O adapter converte `conversationId` em `configurable.thread_id`. Repetir o ID
continua a conversa; usar outro ID inicia um estado isolado. O checkpointer
atual e local ao processo, conforme detalhado em [`memory.md`](memory.md).

## Eventos publicos

### `status`

Indica uma mudanca visivel de etapa e sempre precede qualquer token:

```text
event: status
data: {"stage":"rag_search","message":"Buscando na base de conhecimento..."}
```

Os stages implementados sao `rag_search`, `query_rewrite` e `generating`.

### `sources`

E emitido uma vez antes do primeiro token:

```text
event: sources
data: {"sources":[]}
```

A lista permanece vazia ate a normalizacao de fontes e citacoes da Etapa 10.
Metadata interna de chunks nao e exposta como contrato temporario.

### `token`

Contem apenas fragmentos da resposta final:

```text
event: token
data: {"text":"Resposta"}
```

O adapter aceita mensagens de assistente somente quando `langgraph_node` e
`generate_answer`. A mensagem humana registrada no estado e tokens de grading,
structured output ou query rewrite sao descartados. Quando o runtime nao
produz chunks incrementais, o adapter envia a resposta terminal inteira como um
unico token de fallback.

### `done`

Encerra uma execucao normal:

```text
event: done
data: {}
```

### `error`

Encerra uma falha depois que o stream HTTP ja foi aberto:

```text
event: error
data: {"code":"stream_failed","message":"Não foi possível concluir a resposta."}
```

O payload nunca inclui stack trace, prompt, resposta interna do grader ou texto
da excecao. Erros de validacao do request ocorrem antes da abertura do stream e
continuam usando a resposta HTTP `422` do FastAPI.

## Headers

A resposta usa `Content-Type: text/event-stream`, `Cache-Control: no-cache` e
`X-Accel-Buffering: no`. O cliente deve processar eventos progressivamente e
considerar a execucao encerrada somente ao receber `done` ou `error`.

## Carregamento do runtime

O adapter recebe uma factory do grafo. Chroma, embedding, reranker e LLM nao sao
carregados durante import da aplicacao nem em `GET /health`; a inicializacao
ocorre dentro do stream da primeira requisicao. Falhas nessa fase tambem sao
convertidas em `error` publico.
