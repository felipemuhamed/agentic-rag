# Upload de documentos pela API

Ate aqui a base so podia ser alimentada pelos CLIs `agentic-rag-ingest` e
`agentic-rag-index-image`. Os endpoints abaixo abrem o mesmo pipeline para o
usuario final, sem dar acesso ao host:

```text
POST /documents            multipart -> arquivo salvo em UPLOAD_DIR
        |                                     |
        v                                     v
job aceito (202)                    loader -> normalizer -> chunker
        |                                     |
        v                                     v
GET /documents/jobs/{id}            embeddings -> Chroma/knowledge
        |
        +-> GET /documents        lista o que esta indexado
        +-> DELETE /documents/{id} remove chunks e arquivo
```

Para o catalogo visual proprietario o caminho e o mesmo, trocando
`/documents` por `/documents/images` e a collection `knowledge` pela
`equipment_images`.

## Por que assincrono

A ingestao carrega o modelo de embedding e processa o documento inteiro: um
manual grande leva de dezenas de segundos a minutos em CPU. A requisicao termina
em `202 Accepted` assim que o arquivo esta salvo e valido, e o processamento
segue em uma fila com `UPLOAD_WORKERS` threads (uma por padrao, porque o
embedding ja satura a CPU). O cliente acompanha pelo job.

## Formatos aceitos

| Endpoint | Extensoes | Limite |
| --- | --- | --- |
| `POST /documents` | `.txt`, `.md`, `.markdown`, `.pdf` | `UPLOAD_MAX_BYTES` |
| `POST /documents/images` | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` | `MULTIMODAL_MAX_IMAGE_BYTES` |

O PDF e lido pela camada de texto com [pypdf](https://pypdf.readthedocs.io/).
Cada pagina vira um cabecalho `## Pagina N`, entao o numero da pagina chega ao
metadado `section` de cada chunk e aparece nas citacoes. PDF digitalizado sem
camada de texto e recusado com `pdf_extraction_failed`: passe OCR antes de
enviar. Nas imagens, os bytes precisam bater com a extensao declarada; a
verificacao usa a mesma assinatura binaria aplicada as imagens do chat.

## Enviar um documento

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'X-API-Key: SUA_CHAVE' \
  -F 'file=@manual-olt.pdf' \
  -F 'tenantId=public' \
  -F 'equipmentId=OLT-9000'
```

```json
{
  "jobId": "1f0c...",
  "status": "queued",
  "statusUrl": "/documents/jobs/1f0c..."
}
```

`equipmentId` e opcional e liga o documento a um equipamento do catalogo visual,
exatamente como a flag `--equipment-id` do CLI.

## Acompanhar o job

```bash
curl 'http://127.0.0.1:8000/documents/jobs/1f0c...?tenantId=public' \
  -H 'X-API-Key: SUA_CHAVE'
```

```json
{
  "jobId": "1f0c...",
  "kind": "document",
  "filename": "manual-olt.pdf",
  "status": "succeeded",
  "createdAt": "2026-08-28T18:00:00+00:00",
  "updatedAt": "2026-08-28T18:00:42+00:00",
  "result": {
    "documentId": "9ac4...",
    "chunksWritten": 128,
    "source": "/data/uploads/public/documents/manual-olt.pdf",
    "filename": "manual-olt.pdf",
    "equipmentId": "OLT-9000"
  },
  "error": null,
  "errorCode": null
}
```

`status` percorre `queued`, `running` e termina em `succeeded` ou `failed`. Em
falha, `errorCode` assume um valor estavel para tratamento no cliente:
`unsupported_document_type`, `pdf_extraction_failed`, `empty_document`,
`invalid_document` ou `ingestion_failed`. O job so e visivel para o mesmo
principal e tenant que o criou, e some depois de
`UPLOAD_JOB_RETENTION_SECONDS`. `GET /documents/jobs` lista os jobs recentes do
solicitante.

## Cadastrar uma foto de referencia

```bash
curl -X POST http://127.0.0.1:8000/documents/images \
  -H 'X-API-Key: SUA_CHAVE' \
  -F 'file=@foto-frontal.png' \
  -F 'equipmentId=OLT-9000' \
  -F 'equipmentName=OLT proprietaria' \
  -F 'manufacturer=CPQD' \
  -F 'model=X9000' \
  -F 'description=Chassi frontal com oito portas opticas' \
  -F 'tenantId=public'
```

Envie varios angulos com o mesmo `equipmentId` para aumentar a cobertura, e
ingira os manuais com esse mesmo identificador para fechar o vinculo
imagem-manual descrito em [`multimodal.md`](multimodal.md).

## Listar e remover

```bash
curl 'http://127.0.0.1:8000/documents?tenantId=public&limit=50&offset=0' \
  -H 'X-API-Key: SUA_CHAVE'

curl -X DELETE 'http://127.0.0.1:8000/documents/9ac4...?tenantId=public' \
  -H 'X-API-Key: SUA_CHAVE'
```

A listagem agrega os chunks por `documentId` e informa `chunkCount`, titulo,
origem, tipo, tenant e `equipmentId`. `GET /documents/images` faz o mesmo na
collection visual. A remocao apaga os chunks e tambem o arquivo em `UPLOAD_DIR`;
`fileRemoved` diz se havia arquivo sob gestao da API (documentos ingeridos pelo
CLI ficam fora dela). Use `DELETE /documents/images/{documentId}` para as fotos.

## Demonstracao pronta

`scripts/demo-rag.sh` executa o roteiro completo com curl, usando os arquivos de
`documents/`. As perguntas nunca citam marca, modelo ou nome do aparelho: a
unica indicacao e a foto anexada ao prompt, e quem identifica o equipamento e o
catalogo visual. O roteiro carrega a especificacao e a foto do primeiro
aparelho, pergunta com essa foto e obtem resposta com fontes; repete a pergunta
com a foto de outro aparelho, ainda sem manual indexado, e recebe a recusa
segura; carrega o segundo manual e repete a mesma pergunta com a mesma foto,
agora respondida. O script comeca removendo os documentos da demo, entao pode
ser repetido a vontade.

```bash
scripts/ambiente.sh subir          # constroi e sobe chroma + api, espera o /ready
scripts/ambiente.sh aquecer        # opcional: carrega os modelos antes da demo
scripts/demo-rag.sh                # com pausas entre os atos
AUTO=1 scripts/demo-rag.sh         # sem pausas
scripts/demo-rag.sh 4              # so o ato 4
scripts/demo-rag.sh --foto-inedita  # nao cadastra a foto usada nas perguntas
scripts/demo-rag.sh --limpar       # so a limpeza
```

`scripts/ambiente.sh` cuida do Docker Compose: valida `docker`, cria o `.env` a
partir do exemplo quando falta, avisa se `LLM_MODEL` ou a chave do provider
estao vazios, reconstroi a imagem da API (obrigatorio, porque os uploads
dependem de `python-multipart` e `pypdf`), sobe `chroma` e `api` e so retorna
quando `/ready` responde. Tambem aceita `status`, `logs`, `parar` (preserva os
volumes) e `limpar` (`down -v`, destrutivo e com confirmacao).

O comando `aquecer` sobe e remove um arquivo temporario de nome unico apenas
para forcar o download dos modelos dentro do container; ele apaga pelo
`documentId` retornado, entao nao toca em nenhum documento real do acervo.

`API_URL`, `TENANT`, `API_KEY` e `DOCS_DIR` sao configuraveis por variavel de
ambiente, e os scripts funcionam tanto da raiz do repositorio quanto de dentro
de `scripts/`. O roteiro exige `MULTIMODAL_ENABLED=true` e
`VISUAL_CATALOG_ENABLED=true`, ja que toda pergunta leva uma foto. O ato que
demonstra a recusa depende de `WEB_SEARCH_ENABLED=false`: com o fallback web
ligado, a pergunta sem documento interno pode ser respondida pela busca externa.

Por padrao a foto usada nas perguntas tambem esta cadastrada no catalogo, o que
torna o reconhecimento deterministico na apresentacao. Com `--foto-inedita`, a
foto da pergunta nao e cadastrada e o reconhecimento passa a depender da
similaridade com o outro angulo e do limiar `VISUAL_MAX_DISTANCE`: mais
demonstrativo, e tambem mais sujeito a calibracao.

## Identidade e reenvio

O arquivo e gravado em `UPLOAD_DIR/<tenant>/documents/<nome>` com um nome
saneado ainda dentro da requisicao, antes de entrar na fila: a espera nao custa
memoria e o `202` ja garante que o arquivo esta em disco. O `document_id` deriva
de tenant + caminho. Reenviar o mesmo nome
substitui a versao anterior no indice em vez de duplicar, e tenants diferentes
nunca compartilham diretorio: identificadores que nao sejam
`[A-Za-z0-9._-]` viram um diretorio derivado de hash, evitando colisao.

## Isolamento e limites

Quando a fila esta cheia o arquivo ja gravado permanece no volume sem ser
indexado; reenviar o mesmo nome depois o sobrescreve. A API nao apaga esse
arquivo automaticamente porque um job pendente pode estar esperando exatamente
por ele.

Os endpoints usam a mesma politica de `X-API-Key` e `tenantId` do chat. Com
`AUTH_ENABLED=true`, listagem e remocao so enxergam o tenant autenticado, e o
filtro tambem vale para a exclusao: tentar remover documento de outro tenant
responde `404`. O rate limit do chat tambem se aplica; com a fila cheia a API
responde `503` com `Retry-After`.

| Variavel | Padrao | Efeito |
| --- | --- | --- |
| `UPLOAD_ENABLED` | `true` | `false` remove os endpoints (`404`) |
| `UPLOAD_DIR` | `./data/uploads` | Raiz dos arquivos enviados |
| `UPLOAD_MAX_BYTES` | `20000000` | Tamanho maximo por documento |
| `UPLOAD_WORKERS` | `1` | Ingestoes simultaneas |
| `UPLOAD_MAX_PENDING_JOBS` | `32` | Fila antes do `503` |
| `UPLOAD_MAX_JOBS` | `200` | Jobs mantidos em memoria |
| `UPLOAD_JOB_RETENTION_SECONDS` | `3600` | Tempo de vida do job terminado |

Os jobs vivem no processo da API: reiniciar o container descarta o historico,
mas nao o que ja foi indexado no Chroma nem os arquivos no volume. No Compose,
`UPLOAD_DIR` aponta para o volume `upload-data`, o unico ponto gravavel do
container alem dos checkpoints.
