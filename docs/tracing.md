# Observabilidade de LLM com Langfuse self-hosted

O `/metrics` responde quanto tempo um stream levou, mas nao o que aconteceu
dentro dele: qual consulta o preparador produziu, o que o grader viu, quais
chunks entraram no prompt e quanto custou cada chamada. O Langfuse cobre essa
camada. Self-hosted, prompts, respostas e imagens analisadas permanecem na
infraestrutura do projeto, o que e o requisito quando o acervo e proprietario.

## O que e instrumentado

Cada requisicao de `POST /chat/stream` abre uma observacao raiz `chat-stream` e
tudo que acontece dentro dela vira filho do mesmo trace:

```text
chat-stream                      (session_id = thread derivado da conversa)
├── analise visual da imagem     (modelo com visao, quando ha imagem)
└── grafo LangGraph
    ├── prepare_query
    ├── grade_rag
    ├── rewrite_query            (quando ha nova tentativa)
    └── generate_answer
```

Sem a observacao raiz, a chamada de visao e a execucao do grafo apareceriam como
dois traces sem relacao para uma unica pergunta. O agrupamento usa
`propagate_attributes`, e o `session_id` e o thread derivado em
[`access.py`](../app/security/access.py) -- um digest de principal, tenant e
`conversationId`. Nenhum identificador publico ou dado pessoal e enviado.

## O tracing nunca derruba o produto

Uma resposta que falha porque o coletor de traces esta fora e um resultado pior
do que uma resposta que ninguem consegue inspecionar. Por isso
[`tracing.py`](../app/observability/tracing.py) degrada em vez de propagar:
pacote ausente, credenciais faltando, cliente que nao constroi ou span que nao
abre resultam em um `NullTracing` silencioso com registro no log. Apenas a
*entrada* do contexto e protegida -- excecoes do proprio request continuam
subindo normalmente, entao um erro real nunca vira sucesso silencioso.

O cliente recebe `public_key`, `secret_key` e `base_url` explicitamente das
configuracoes do projeto, em vez de deixar o SDK ler o ambiente. O SDK renomeou
a variavel de host entre versoes maiores, e passar o valor direto evita que o
deploy dependa de qual nome a versao instalada honra.

## Subir a stack

Sao seis containers a mais (web, worker, Postgres, ClickHouse, Redis e MinIO),
atras do profile `observability`, entao nada disso sobe no `docker compose up -d
api` do dia a dia. Reserve por volta de 4 vCPU e 16 GiB para a stack completa.

Gere os segredos e coloque no `.env`:

```bash
echo "LANGFUSE_SALT=$(openssl rand -base64 32)"        >> .env
echo "LANGFUSE_NEXTAUTH_SECRET=$(openssl rand -base64 32)" >> .env
echo "LANGFUSE_ENCRYPTION_KEY=$(openssl rand -hex 32)" >> .env
```

O `ENCRYPTION_KEY` precisa ter exatamente 256 bits, ou seja 64 caracteres
hexadecimais; `openssl rand -base64 32` **nao** serve para ele.

Defina tambem as chaves de projeto, que sao usadas em dois papeis: o servidor as
cadastra no primeiro boot e o app autentica com elas.

```dotenv
LANGFUSE_PUBLIC_KEY=lf_pk_...
LANGFUSE_SECRET_KEY=lf_sk_...
LANGFUSE_INIT_USER_EMAIL=voce@cpqd.com.br
LANGFUSE_INIT_USER_PASSWORD=uma-senha-forte
```

Os prefixos `lf_pk_` e `lf_sk_` sao exigidos pelo Langfuse v4. Entao:

```bash
scripts/ambiente.sh langfuse       # equivale ao comando abaixo, com avisos
docker compose --profile observability up -d
```

A UI fica em `http://localhost:3000`. O primeiro boot roda migracoes de Postgres
e ClickHouse e demora; acompanhe com `docker compose logs -f langfuse-web`.

## Ligar no app

```bash
sed -i 's/^LANGFUSE_ENABLED=.*/LANGFUSE_ENABLED=true/' .env
docker compose up -d --force-recreate api
```

Confirme no log do `api` a linha `Langfuse tracing enabled`. Se aparecer
`continuing without tracing`, a mensagem diz qual das tres causas ocorreu:
pacote ausente, credencial faltando ou cliente que nao construiu.

"Pacote ausente" tem uma causa historica que vale registrar: o `Dockerfile`
construia o wheel do extra `tracing` mas instalava so `[knowledge,rag,web]`, e a
imagem subia sem `langfuse`. Um wheel em `/wheels` que ninguem instala nao esta
na imagem; a linha de instalacao tem de nomear o extra.

Fora do Compose, aponte `LANGFUSE_BASE_URL` para `http://localhost:3000` -- o
default `http://langfuse-web:3000` so resolve dentro da rede do Compose.

## Versoes

A stack segue a composicao oficial da v4: imagens `langfuse:4` e
`langfuse-worker:4`, ClickHouse 25.12 (a v4 exige >= 25.12), Redis 7 e Postgres
17. O SDK Python fica em `langfuse>=4.7`, primeira versao cuja ingestao o
servidor v4 aceita, e entra pelo extra `tracing`.

Postgres e ClickHouse rodam com `TZ: UTC` de proposito. Fora de UTC, as consultas
do Langfuse retornam resultados incorretos ou vazios, e a falha e silenciosa.

## Limites desta entrega

Os spans cobrem as chamadas de LLM porque o handler e um callback do LangChain.
Retrieval no Chroma, reranker e embedding de imagem nao aparecem como spans
proprios: eles nao passam pelo callback do LangChain. Instrumenta-los pede spans
manuais no retriever, o que e uma iteracao separada.

Custo por resposta so aparece se o modelo devolver contagem de tokens pelo
OpenRouter. Quando nao devolver, o trace mostra latencia e conteudo, mas nao
preco.
