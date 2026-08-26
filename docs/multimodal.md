# Consulta e catalogo visual multimodal

As Etapas 13 e 14 aceitam uma imagem junto da pergunta, analisam seu conteudo e
comparam sua representacao vetorial com fotos de equipamentos proprietarios:

```text
imagem Base64 + pergunta
        |
        v
analise visual estruturada (resumo, OCR, identificadores, incertezas)
        |
        +-> documento visual transitorio e citavel
        |
        +-> embedding visual -> Chroma/equipment_images
                                |
                                v
                  candidatos + equipment_id
                                |
        +-> consulta textual + filtro de equipamento
                                |
                                v
              Chroma/knowledge -> MMR -> reranker
                                |
                                v
                       grade -> resposta com fontes
```

O modelo configurado precisa aceitar imagem. No OpenRouter, Claude Sonnet 4.6
aceita texto, imagens e arquivos como entrada. O `ChatOpenRouter` recebe texto e
imagem em uma `HumanMessage` com lista de blocos. A implementacao segue as
documentacoes oficiais de [imagem no OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding)
e [entrada multimodal no LangChain](https://docs.langchain.com/oss/python/integrations/chat/openrouter#multimodal-inputs).

O catalogo usa um modelo CLIP do Sentence Transformers, que posiciona imagens
e texto no mesmo espaco vetorial e suporta image-to-image search. Consulte a
[documentacao multimodal do Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
e o [model card do CLIP ViT-B/32](https://huggingface.co/sentence-transformers/clip-ViT-B-32).
A collection visual e separada porque uma query Chroma precisa ter a mesma
dimensao dos embeddings da collection, conforme o
[contrato de query do Chroma](https://docs.trychroma.com/docs/querying-collections/query-and-get).

## Contrato HTTP

```json
{
  "conversationId": "imagem-1",
  "tenantId": "public",
  "message": "Que equipamento e este e qual procedimento se aplica?",
  "images": [
    {
      "name": "painel.png",
      "mimeType": "image/png",
      "data": "BASE64_SEM_PREFIXO_DATA_URL"
    }
  ]
}
```

Formatos aceitos: `image/png`, `image/jpeg`, `image/webp` e `image/gif`. O
servidor valida Base64, assinatura binaria, quantidade e tamanho antes de
reservar a execucao do stream. URLs externas nao sao aceitas nesta etapa, o que
evita transformar o backend em um cliente HTTP arbitrario.

## Configuracao

```dotenv
MULTIMODAL_ENABLED=true
VISION_MODEL=
MULTIMODAL_MAX_IMAGES=1
MULTIMODAL_MAX_IMAGE_BYTES=5000000
VISUAL_CATALOG_ENABLED=true
VISUAL_COLLECTION=equipment_images
VISUAL_EMBEDDING_MODEL=sentence-transformers/clip-ViT-B-32
VISUAL_EMBEDDING_DEVICE=cpu
VISUAL_MAX_RESULTS=3
VISUAL_MAX_DISTANCE=0.8
```

`VISION_MODEL` vazio reutiliza `LLM_MODEL`. Um modelo separado permite escolher
um vision model mais barato sem alterar graders e gerador. Os limites absolutos
do contrato sao cinco imagens e 20 MB decodificados por imagem; os defaults de
deploy sao mais restritos.

`VISUAL_MAX_DISTANCE` e um gate: referencias acima dessa distancia nao entram
na resposta nem filtram os manuais. O valor `0.8` e apenas inicial e deve ser
calibrado no catalogo real com pares positivos (mesmo equipamento) e negativos
(equipamentos parecidos).

## Cadastro de equipamentos proprietarios

Uma referencia visual e seus documentos tecnicos precisam usar exatamente o
mesmo `equipment_id`. Primeiro disponibilize os arquivos no diretorio montado
como `/documents` e execute:

```bash
docker compose run --rm index-image \
  --equipment-id OLT-9000 \
  --equipment-name "OLT proprietaria" \
  --manufacturer CPQD \
  --model X9000 \
  --description "Vista frontal do chassi com oito portas" \
  --tenant-id public \
  /documents/olt-frontal.png /documents/olt-traseira.jpg

docker compose run --rm ingest \
  --equipment-id OLT-9000 \
  --tenant-id public \
  /documents/manual-olt.md
```

Cada foto e uma referencia independente. Fotos de frente, traseira, etiqueta,
placa e diferentes condicoes de iluminacao aumentam a chance de recuperacao.
O catalogo armazena embedding, metadados, hash e caminho de origem; nao copia o
binario da imagem para o checkpointer nem para a collection.

Na consulta, o filtro de tenant tambem e aplicado ao catalogo visual. Portanto,
o mesmo `tenant_id` deve ser usado no cadastro das fotos, na ingestao dos
manuais e no request HTTP.

## Seguranca e persistencia

Texto dentro da imagem e tratado como dado e nunca como instrucao. A analise
separa observacao de incerteza e nao deve completar trechos ilegiveis. O Base64
existe somente antes das chamadas visual e vetorial e nao entra no
state/checkpointer do LangGraph. O estado persiste apenas texto analisado,
referencias recuperadas, hash e nome, sem os bytes da imagem.

A analise visual vira uma fonte RAG com locator `user-image:<hash>`. Ela pode
ser citada na resposta ao lado das referencias do catalogo e dos chunks dos
manuais. Follow-ups reutilizam a analise textual registrada no historico, sem
reenviar nem armazenar o arquivo.

## Limites e confianca

Similaridade visual produz candidatos, nao uma identidade infalivel. A resposta
recebe a distancia e e instruida a confrontar a referencia, a analise visual e
os manuais. Em producao, avalie top-k, falso positivo, falso negativo e um
limiar de abstencao por familia de equipamento.

Esta entrega indexa fotos explicitas de referencia. Extracao automatica de
figuras de PDFs, recortes por regiao, armazenamento duravel do objeto original
e modelos treinados no dominio continuam como evolucoes separadas.
