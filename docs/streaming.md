# Streaming SSE para o frontend

A Etapa 7 expoe um protocolo publico estavel e mantem o formato nativo de
streaming do LangGraph dentro de `app/streaming`.

## Endpoint

```http
POST /chat/stream
Content-Type: application/json
Accept: text/event-stream
X-API-Key: <chave, quando AUTH_ENABLED=true>

{
  "conversationId": "demo-1",
  "tenantId": "public",
  "message": "Como renovar o token?"
}
```

`conversationId` e `message` sao obrigatorios, textuais e nao podem conter
apenas espacos. `tenantId` e obrigatorio quando a autenticacao esta ativa. A
mensagem aceita ate 8.000 caracteres e os IDs ate 128. Campos desconhecidos sao
rejeitados. O thread interno combina principal, tenant e conversa, conforme
detalhado em [`memory.md`](memory.md).

## Eventos publicos

### `status`

Indica uma mudanca visivel de etapa e sempre precede qualquer token:

```text
event: status
data: {"stage":"rag_search","message":"Buscando na base de conhecimento..."}
```

Os stages implementados sao `rag_search`, `query_rewrite`, `web_search` e
`generating`. `web_search` so aparece depois que o limite do RAG interno e
atingido.

### `sources`

E emitido uma vez antes do primeiro token:

```text
event: sources
data: {"sources":[{"id":1,"citation":"[1]","type":"rag","title":"Guia OAuth","source":"oauth.md","chunkId":"oauth-1","url":null,"page":3,"section":"Tokens"}]}
```

A lista contem somente fontes citadas na resposta terminal. Para RAG, cada item
mapeia para um `chunkId`; para web, mapeia para uma URL. Campos indisponiveis
sao enviados como `null`, mantendo um contrato estavel. Consulte
[`citations.md`](citations.md).

### `token`

Contem apenas fragmentos da resposta final:

```text
event: token
data: {"text":"Resposta"}
```

O adapter nao encaminha diretamente nenhum token do runtime interno. Grading,
rewrite e a geracao permanecem privados ate a resposta terminal passar pela
validacao de citacoes. Depois disso, o adapter emite `sources` e fragmenta a
resposta validada em eventos `token`. Assim, uma referencia inexistente produz
`error` antes que qualquer fonte ou texto final seja exibido.

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

Um deadline total excedido usa `code=stream_timeout`. Cancelamento pelo cliente
encerra a iteracao sem evento terminal, pois a conexao ja nao esta disponivel.
O payload nunca inclui stack trace, prompt, resposta interna do grader ou texto
da excecao. Erros de validacao do request ocorrem antes da abertura do stream e
continuam usando a resposta HTTP `422` do FastAPI.

Falha de autenticacao retorna HTTP `401`, tenant nao autorizado retorna `403` e
quota esgotada retorna `429` com `Retry-After`, todos antes da abertura do SSE.

## Headers

A resposta usa `Content-Type: text/event-stream`, `Cache-Control: no-cache` e
`X-Accel-Buffering: no`. Inclui tambem `X-Request-ID`, `X-RateLimit-Limit` e
`X-RateLimit-Remaining`. O cliente deve processar eventos progressivamente e
considerar a execucao encerrada somente ao receber `done` ou `error`.

## Carregamento do runtime

O adapter recebe uma factory do grafo. Chroma, embedding, reranker e LLM nao sao
carregados durante import da aplicacao nem em `GET /health`; a inicializacao
ocorre dentro do stream da primeira requisicao. Falhas nessa fase tambem sao
convertidas em `error` publico e fazem `/ready` responder `503` ate que uma
inicializacao posterior seja bem-sucedida. Os controles completos estao em
[`production.md`](production.md).
