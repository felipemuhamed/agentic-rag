# Citacoes e proveniencia

A Etapa 10 torna cada referencia da resposta rastreavel ate o chunk interno ou
a URL externa que sustentou a afirmacao.

## Modelo publico

O evento SSE `sources` usa um formato unico para as duas origens:

```json
{
  "id": 1,
  "citation": "[1]",
  "type": "rag",
  "title": "Manual de autenticacao",
  "source": "manual-oauth.pdf",
  "chunkId": "chunk-oauth-7",
  "url": "https://docs.example.com/manual-oauth.pdf",
  "page": 7,
  "section": "Configuracao OAuth"
}
```

Os campos sao:

- `id` e `citation`: identificador numerico e rotulo exibido na resposta;
- `type`: `rag` ou `web`;
- `title` e `source`: nome e localizador original;
- `chunkId`: obrigatorio para RAG e `null` para web;
- `url`: obrigatorio para web e preservado quando existir no RAG;
- `page` e `section`: preservados da metadata interna quando disponiveis.

Campos opcionais permanecem no payload como `null`, evitando formatos
diferentes por provider.

## Atribuicao de IDs

Depois de um grade positivo, as evidencias aprovadas sao numeradas na ordem em
que chegam ao gerador. O contexto entregue ao LLM contem `citation_id` e o
rotulo canonico correspondente. O prompt exige pelo menos uma citacao e proibe
qualquer ID que nao esteja no contexto.

Uma fonte RAG mapeia para um chunk exato. Uma fonte web mapeia para a URL exata
retornada pelo `WebSearchProvider`. A lista publica final remove evidencias que
nao foram citadas, sem renumerar os IDs usados no texto.

## Validacao fechada

A aplicacao nao depende apenas do prompt. Antes de concluir `generate_answer`:

1. extrai todos os rotulos no formato `[n]`;
2. exige ao menos um rotulo para uma resposta grounded;
3. rejeita IDs ausentes da lista de fontes;
4. rejeita variantes nao canonicas, como `[01]`;
5. mantem apenas as fontes efetivamente citadas.

O adapter SSE repete a validacao na fronteira publica. Ele exige IDs unicos,
tipo coerente com `source=rag|web` e correspondencia exata entre citacoes e a
lista `sources`. Qualquer inconsistencia gera o evento seguro `error` antes de
emitir fontes ou tokens.

## Ordem dos eventos

Para impedir que uma citacao inventada apareca durante o streaming nativo do
modelo, mensagens internas nao sao encaminhadas diretamente. Status continuam
sendo emitidos durante a execucao; apos a validacao terminal, o adapter envia:

```text
sources -> token -> token -> ... -> done
```

Os tokens sao fragmentos da resposta final ja validada. Essa escolha prioriza
consistencia e auditabilidade: o primeiro token aguarda a conclusao da geracao e
da verificacao de referencias.

## Limites

A etapa garante que toda citacao exibida aponta para uma fonte existente. Ela
nao prova automaticamente a qualidade semantica da atribuicao; graders e prompt
continuam responsaveis pelo grounding. A Etapa 11 adicionou proxies
reproduziveis e revisao por caso; controles de acesso a documentos/fontes
pertencem ao hardening da Etapa 12.
