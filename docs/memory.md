# Threads e memoria conversacional

A Etapa 8 adicionou continuidade entre chamadas de `POST /chat/stream` sem
misturar historico de conversa com a base de conhecimento. A Etapa 12 tornou
essa memoria duravel e associou os threads a identidade autorizada.

## Identidade do thread

O contrato HTTP recebe `conversationId`. Sem autenticacao, o adapter ainda pode
usar esse valor diretamente em desenvolvimento. Com `AUTH_ENABLED=true`, a API
autoriza `X-API-Key` e `tenantId` e calcula:

```text
thread_id = sha256(principal + tenant + conversationId)
```

Somente esse digest e enviado como `configurable.thread_id`. Assim,
`conversationId` iguais de tenants ou principals diferentes possuem estados
independentes e o identificador publico nao e gravado diretamente no
checkpointer.

## Preparacao da consulta

O estado preserva as mensagens humana e de assistente de cada turno concluido.
Antes do retrieval, `prepare_query` recebe a mensagem atual e o historico do
thread. No primeiro turno, a mensagem e preservada sem chamar o modelo. Em um
follow-up, o preparer resolve apenas referencias, elipses e contexto implicito:

```text
historico:  "Como configurar OAuth?"
mensagem:   "E no Keycloak?"
prepared:   "Como configurar OAuth no Keycloak?"
```

`prepared_query` e a intencao estavel do turno. `search_query` comeca com esse
valor e pode mudar durante os retries. O texto original permanece em
`question`, para que o par de mensagens represente o que o usuario escreveu.

## Fronteira de contexto

O historico bruto e acessado somente pelo query preparer. Retrieval, grader,
query rewrite e geracao recebem a consulta autocontida, e nao toda a conversa.
Isso reduz acoplamento, consumo de contexto e o risco de instrucoes antigas
afetarem gates que nao precisam delas.

Chroma continua armazenando documentos e embeddings da base de conhecimento.
Mensagens e respostas nao sao indexadas nem gravadas no Chroma. O tenant da
conversa e aplicado separadamente como filtro de retrieval.

## Backends de checkpoint

`CHECKPOINTER_MODE=memory` usa `InMemorySaver`: e rapido para testes, mas perde
todos os threads ao reiniciar e nao compartilha estado entre processos.

`CHECKPOINTER_MODE=sqlite` usa um arquivo SQLite com WAL, espera para locks e
serializacao estrita. O Compose monta `checkpoint-data` no caminho do arquivo,
por isso recriar o container da API preserva as conversas. Esse modo atende uma
instancia; replicas em hosts distintos exigem um saver compartilhado, como
PostgreSQL.

O produto ainda precisa definir retencao, exclusao por titular e backup
conforme seus requisitos de privacidade. A CLI cria um `thread_id` unico por
execucao e permanece intencionalmente single-turn. Consulte
[`production.md`](production.md) para a operacao.

## Criterios automatizados

Os testes verificam que:

1. um primeiro turno nao chama o preparer via LLM;
2. "E no Keycloak?" vira uma consulta OAuth autocontida no mesmo thread;
3. outro ID nao recebe o historico anterior;
4. os checkpoints isolam suas mensagens;
5. mensagens humanas nao sao emitidas como tokens SSE;
6. o estado SQLite sobrevive ao fechamento e reabertura do runtime;
7. principal e tenant diferentes produzem IDs de thread distintos.
