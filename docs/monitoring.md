# Dados ao vivo da monitoracao via MCP

Perguntas sobre alarmes nao sao perguntas de documentacao. "Quais os ultimos
alertas?" nao tem resposta no acervo, por mais completo que ele seja: a resposta
esta no Zabbix, e muda a cada minuto. Esta entrega faz o agente consultar um
servidor MCP de monitoracao quando -- e somente quando -- a pergunta pede isso.

```text
START
  |
  +-- classificador: a pergunta precisa de dado ao vivo?
        |                                   |
        | sim                               | nao
        v                                   |
   search_monitoring                        |
   (loop de tool-calling limitado)          |
        |                                   |
        +-----------> prepare_query <-------+
                            |
                      retrieve_rag -> grade_rag -> ... -> generate_answer
```

A monitoracao roda **antes** do `prepare_query` de proposito. E o alarme que
determina o que procurar nos manuais: sabendo que o evento e "perda optica alta"
no host `olt-sp-01`, a consulta que busca a documentacao deixa de ser "alertas
criticos" e passa a conter o sintoma e o equipamento. Se rodasse depois, a busca
textual ja teria acontecido com a pergunta vaga.

## Tres perguntas, um caminho

| Pergunta | O que acontece |
|---|---|
| "Quais os ultimos alertas?" | So monitoracao. As observacoes bastam para o grader e viram a resposta. |
| "Como reinicio a OLT?" | So documentacao. O classificador nem aciona a monitoracao. |
| "Explique os alertas mais criticos" | Os dois. Os eventos entram como evidencia e alimentam a busca dos manuais dos equipamentos afetados. |

O terceiro caso e o que justifica a entrega, e e onde o resolvedor de equipamento
da Etapa 17 se paga: o host do alarme vira `equipment_id`, que filtra o manual
certo.

## Evidencia citavel, nao contexto solto

Cada chamada de ferramenta vira um `RetrievedDocument` com locator
`zabbix:<ferramenta>`, carregando a ferramenta e os argumentos que a produziram.
Isso mantem o contrato de citacao do servico: o operador ve que a afirmacao veio
de `problem_get(limit=10)` e nao da memoria do modelo, e consegue reproduzir a
chamada. Os documentos ao vivo lideram a lista de evidencias porque sao a
premissa da pergunta, nao um acerto de similaridade.

## Allowlist e o limite de seguranca

Um servidor MCP completo de Zabbix publica a API inteira, incluindo escrita em
hosts, itens e templates. **Uma pergunta digitada num chat nao pode alcancar
isso.** Por isso `MONITORING_ALLOWED_TOOLS` e a fronteira: apenas as ferramentas
nomeadas ali sao entregues ao modelo. Ampliar o que o agente pode fazer e uma
mudanca de configuracao explicita, nunca consequencia do servidor para o qual ele
aponta.

Vazio significa nenhuma ferramenta, e o log avisa. Descubra os nomes reais:

```bash
agentic-rag-monitoring-tools --url http://zabbix-mcp:8080/mcp --filter problem
```

Depois preencha, por exemplo:

```dotenv
MONITORING_ENABLED=true
MONITORING_MCP_URL=http://zabbix-mcp:8080/mcp
MONITORING_MCP_TOKEN=<token de leitura>
MONITORING_ALLOWED_TOOLS=problem_get,trigger_get,event_get,host_get,history_get
```

Use um token **somente leitura** aqui, mesmo com a allowlist. Sao duas defesas
independentes: a allowlist limita o que o modelo enxerga, o token limita o que o
servidor aceita executar. Escrita, se for o caso, pertence a uma sessao com um
humano lendo cada passo -- nao ao chat automatico.

## Limites de execucao

`MONITORING_MAX_ITERATIONS` (default 3) limita as voltas do loop de
tool-calling; atingido o teto, o agente para com o que ja coletou.
`MONITORING_TIMEOUT_SECONDS` corta uma chamada travada, cancelando a tarefa em
vez de deixa-la ocupando o loop.

Uma ferramenta que falha e **registrada como observacao de erro**, nao levantada.
O gerador precisa ver que o dado faltou; o contrario seria uma resposta confiante
construida sobre nada. E se a monitoracao inteira estiver fora, o no degrada para
documentacao apenas: uma pergunta hibrida ainda recebe a parte que os manuais
respondem.

## Tres resultados, nao dois

Uma consulta ao Zabbix pode terminar de tres formas, e trata-las igual produz
respostas erradas. Cada observacao carrega um status, e o no resume o conjunto:

| Status | O que aconteceu | Consequencia |
|---|---|---|
| `ok` | Ao menos uma chamada trouxe dados | Fluxo normal: RAG e web podem enriquecer |
| `empty` | As chamadas rodaram e voltaram vazias | Sem reescrita e sem web |
| `unavailable` | Nenhuma chamada chegou a responder | Sem reescrita e sem web |

**Vazio nao e falta de evidencia: e um fato.** `problem_get` respondendo `[]`
significa que nao ha alarmes, e isso responde a pergunta. Por isso a observacao
vazia continua virando documento citavel, com o corpo escrito em palavras
("Consulta executada. Nenhum resultado retornado.") em vez de um `[]` que o
gerador leria como ausencia de contexto.

O que **nao** pode acontecer e cair no fallback. Se a pergunta passou pela
monitoracao e nao veio dado, reescrever a consulta ou buscar na web seria
maquiar um dado ausente: nenhum manual e nenhuma pagina da internet sabe o
estado atual desta infraestrutura. O roteamento pos-grader corta os dois
caminhos e a resposta diz qual dos dois nadas aconteceu -- "consultei e nao
retornou nada" ou "nao consegui consultar" --, que sao situacoes diferentes para
quem opera. O enriquecimento continua valendo quando o status e `ok`: os alarmes
encontrados seguem alimentando a busca dos manuais.

## Demonstrar sem um Zabbix por perto

Um comando faz o caminho inteiro, do zero:

```bash
scripts/ambiente.sh monitoracao
```

Ele sobe a API se preciso, reconstroi a imagem se ela estiver sem os extras,
sobe o servidor MCP falso, escreve as chaves `MONITORING_*` no `.env`, recria a
API para que ela as leia -- variavel de ambiente e lida na criacao do container,
e esquecer isso e o erro mais comum aqui --, lista as ferramentas publicadas e
so entao roda o roteiro. Com `--sem-env` ele nao toca no `.env` e apenas mostra
o que faltaria configurar.

O roteiro, `scripts/demo-monitoracao.sh`, roda quatro atos: pergunta de procedimento (a
monitoracao nem e acionada), pergunta de alarmes (aparece o status
`monitoring_search` e as fontes marcadas como ao vivo), pergunta hibrida (alarme
e manual na mesma resposta) e consulta a um host inexistente, que e o ato que
importa: o Zabbix responde lista vazia e a resposta diz que nao encontrou, sem
reescrever a busca nem cair na web.

Numa maquina sem Zabbix, `scripts/ambiente.sh zabbix-falso` sobe
`scripts/zabbix-falso.py`, um servidor MCP de **mentira** com dados inventados.
Ele publica `host_update` de proposito: e a allowlist, e nada mais, que mantem a
escrita fora do alcance do chat, e isso so fica visivel quando a ferramenta
existe do outro lado.

Antes de montar a allowlist, veja o que o servidor publica de fato:

```bash
scripts/ambiente.sh ferramentas --filter problem
docker compose run --rm monitoring-tools --filter problem   # equivalente
```

Os dois servicos novos do Compose ficam fora do caminho do dia a dia:
`monitoring-tools` no profile `tools`, junto de `ingest`, `retrieve` e `rag`; e
`zabbix-falso` num profile `demo` proprio, para que nenhum `up -d` distraido o
suba ao lado de dados reais. `scripts/ambiente.sh parar` usa `--remove-orphans`
justamente para derrubar tambem o que subiu por perfil.

### 421 Misdirected Request

Se o cliente receber `421` do servidor MCP, o servidor esta recusando o header
`Host`. O SDK valida `Host` contra uma allowlist para barrar DNS rebinding, e a
lista default so aceita `localhost` e `127.0.0.1` -- dentro do Compose o cliente
chama `http://zabbix-falso:8080/mcp`, um nome que precisa estar nomeado. No
servidor falso isso ja esta configurado, e `--allow-host` acrescenta outros. Num
Zabbix MCP de verdade, a configuracao equivalente fica do lado dele.

## A ponte sincrona

O pipeline e inteiramente sincrono e o cliente MCP e async-only -- as ferramentas
do adapter levantam `NotImplementedError` em `invoke()` e so atendem `ainvoke()`.
Em vez de converter o pipeline, `app/monitoring/bridge.py` mantem um event loop
proprio numa thread daemon e submete corrotinas a ele com
`run_coroutine_threadsafe`.

O transporte recomendado e Streamable HTTP. O adapter abre uma sessao por
chamada; sobre stdio isso significaria subir um processo do servidor a cada
consulta.

## Dependencia

O extra `monitoring` traz `langchain-mcp-adapters`, que **exige `mcp<2.0.0`**.
Se algum dia este projeto voltar a precisar do SDK v2, os dois nao convivem no
mesmo ambiente.

O extra precisa estar na linha de **instalacao** do `Dockerfile`, nao apenas na
de construcao dos wheels: um wheel presente em `/wheels` que ninguem instala
deixa a feature ausente da imagem, e o `MonitoringToolset` degrada em silencio
como se o servidor estivesse fora.
