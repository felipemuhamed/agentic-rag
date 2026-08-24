# Avaliacao reproduzivel

A Etapa 11 fornece um runner offline separado da API e do LangGraph. Ele mede
retrieval, ranking e respostas sobre uma revisao exata de dataset e corpus. O
relatorio registra parametros, modelos, latencia, uso de tokens e precos
informados, permitindo comparar uma mudanca por vez.

## Dataset inicial

O arquivo `evaluation/datasets/project-docs-v1.json` contem seis perguntas de
manutencao sobre contratos reais deste repositorio. Cada caso declara:

- `id` e `question`;
- um ou mais `expected_documents`, identificados por `document_id` ou `source`;
- `reference_answer` para a avaliacao lexical da resposta;
- tags opcionais e filtros de metadata.

O campo `corpus` enumera todos os arquivos da revisao. O loader rejeita fontes
esperadas que nao pertencam a essa lista. Dataset e conteudo dos arquivos
recebem fingerprints SHA-256 diferentes no relatorio; por isso, relatorios de
revisoes distintas nao podem ser comparados por engano.

Esse conjunto inicial mede regressao do software com dados reais do projeto,
mas nao representa perguntas de usuarios finais de uma operadora. Antes de
calibrar os modelos para producao, ele deve ser ampliado com perguntas e
documentos representativos do dominio, revisados por especialistas.

## Metricas

As metricas de retrieval operam sobre documentos, nao IDs instaveis de chunks:

- `Recall@K`: fracao dos documentos esperados encontrada nos primeiros K
  chunks;
- `MRR`: media do inverso da primeira posicao que pertence a um documento
  esperado;
- latencia media, p50 e p95.

A avaliacao de respostas e deterministica e sem um segundo LLM julgador:

- `relevance`: F1 lexical entre resposta candidata e resposta de referencia;
- `grounding`: proporcao dos tokens relevantes da resposta presentes nos
  chunks efetivamente citados;
- `citation_validity`: validade estrutural e correspondencia entre texto e
  fontes;
- `expected_source_recall`: fracao dos documentos esperados citada;
- latencia, chamadas, tokens e custo estimado quando os precos forem fornecidos.

As metricas lexicais sao proxies conservadoras: uma parafrase correta pode
receber nota menor e sobreposicao de palavras nao prova entailment. Os casos
individuais permanecem no relatorio para revisao humana. Um judge semantico so
deve ser adicionado com prompt, modelo e versao fixos.

## Baseline local deterministico

O modo `--offline` usa hashing vetorial, MMR, reranking por sobreposicao de
tokens e uma resposta extrativa. Esses componentes existem somente no modulo
de avaliacao; o fluxo de producao continua usando os providers configurados.

Execute o baseline sem Chroma, downloads ou credenciais:

```bash
agentic-rag-evaluate evaluation/datasets/project-docs-v1.json \
  --offline \
  --name offline-project-docs-v1 \
  --output evaluation/baselines/offline-project-docs-v1.json
```

O baseline versionado e um gate reproduzivel do framework. Ele nao deve ser
interpretado como qualidade dos modelos BGE ou como calibracao no corpus final.

## Avaliar a configuracao real

Use uma collection isolada e ingira exatamente o corpus do dataset com os
mesmos parametros de chunking registrados no relatorio:

```bash
export CHROMA_COLLECTION=agentic_rag_eval_project_v1
agentic-rag-ingest \
  docs/citations.md \
  docs/containers.md \
  docs/memory.md \
  docs/query-rewrite.md \
  docs/retrieval.md \
  docs/web-fallback.md

agentic-rag-evaluate evaluation/datasets/project-docs-v1.json \
  --name bge-default-v1 \
  --output evaluation/reports/bge-default-v1.json \
  --baseline evaluation/baselines/offline-project-docs-v1.json
```

Sem `--retrieval-only`, o comando exige `LLM_MODEL` e executa a geracao com
temperatura zero por padrao. Para medir apenas retrieval e ranking:

```bash
agentic-rag-evaluate evaluation/datasets/project-docs-v1.json \
  --retrieval-only \
  --name bge-retrieval-v1 \
  --output evaluation/reports/bge-retrieval-v1.json
```

Precos nao sao embutidos, pois variam por provider e data. Quando aplicavel,
informe os valores auditados por milhao de tokens:

```bash
agentic-rag-evaluate evaluation/datasets/project-docs-v1.json \
  --input-cost-per-million 0.00 \
  --output-cost-per-million 0.00
```

O relatorio marca uso de tokens como incompleto quando o provider nao devolve
metadata de usage; nesse caso, custo permanece `null` em vez de ser inventado.

## Comparacao controlada

Altere uma configuracao por vez, use outra collection quando o chunking mudar,
reingira o corpus e passe `--baseline`. A comparacao so e aceita quando os
hashes de dataset/corpus e os valores de K coincidem. O JSON resultante mostra
deltas de qualidade, latencia, tokens, custo e todos os campos de configuracao
alterados.

Modelos nao devem ser promovidos apenas por preferencia. Registre no relatorio
o ganho observado, a latencia e o custo da mesma revisao de dados antes de
alterar os defaults da aplicacao.
