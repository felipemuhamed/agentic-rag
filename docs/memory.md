# Threads e memoria conversacional

A Etapa 8 adiciona continuidade entre chamadas de `POST /chat/stream` sem
misturar historico de conversa com a base de conhecimento.

## Identidade do thread

O contrato HTTP continua recebendo `conversationId`. No limite entre o adapter
SSE e o LangGraph, o valor e passado como:

```python
config={"configurable": {"thread_id": conversation_id}}
```

O grafo e criado uma vez por processo da API e compilado com
`InMemorySaver`. Duas chamadas com o mesmo ID recuperam o checkpoint mais
recente; IDs diferentes possuem estados independentes.

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
`question`, para que o par de mensagens persistido represente o que o usuario
realmente escreveu.

## Fronteira de contexto

O historico bruto e acessado somente pelo query preparer. Retrieval, grader,
query rewrite e geracao recebem a consulta autocontida, e nao toda a conversa.
Isso reduz acoplamento, consumo de contexto e o risco de instrucoes antigas
afetarem gates que nao precisam delas.

Chroma continua armazenando documentos e embeddings da base de conhecimento.
Mensagens de usuario e respostas nao sao indexadas nem gravadas no Chroma.

## Limites do ambiente de desenvolvimento

`InMemorySaver` nao oferece durabilidade nem compartilhamento entre processos:

- reiniciar a API perde todos os threads;
- duas replicas mantem memorias diferentes;
- nao ha TTL, exclusao por usuario ou administracao de checkpoints;
- o `conversationId` ainda nao e associado a uma identidade autenticada.

Esses pontos exigem um checkpointer duravel e controles de seguranca antes de
producao, previstos na Etapa 12. A CLI cria um `thread_id` unico por execucao e,
portanto, permanece intencionalmente single-turn.

## Criterios automatizados

Os testes da etapa verificam que:

1. um primeiro turno nao chama o preparer via LLM;
2. "E no Keycloak?" vira uma consulta OAuth autocontida no mesmo thread;
3. a mesma mensagem em outro ID nao recebe o historico anterior;
4. os checkpoints dos dois IDs contem somente suas proprias mensagens;
5. mensagens humanas persistidas nao sao emitidas como tokens SSE.
