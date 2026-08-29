#!/usr/bin/env bash
# Demonstracao ponta a ponta da RAG multimodal: o aparelho e identificado pela
# FOTO enviada no prompt, nunca pelo texto da pergunta.
#
#   1. Carrega a especificacao do iPhone Air e sua foto de referencia
#   2. Pergunta com a foto do iPhone    -> identifica pela imagem e responde
#   3. Pergunta com a foto do Moto G    -> sem manual indexado, recusa segura
#   4. Carrega a especificacao do Moto G e suas fotos de referencia
#   5. Repete a pergunta com a mesma foto -> agora identifica e responde
#
# Nenhuma pergunta cita marca, modelo ou nome do aparelho: a unica indicacao e a
# imagem anexada ao prompt.
#
# Uso:
#   scripts/demo-rag.sh                 # roteiro completo, com pausas
#   AUTO=1 scripts/demo-rag.sh          # sem pausas
#   scripts/demo-rag.sh --foto-inedita  # nao cadastra a foto usada nas perguntas
#   scripts/demo-rag.sh --limpar        # apenas remove os documentos da demo
#   scripts/demo-rag.sh 3               # executa somente o ato 3
#
# Variaveis: API_URL, TENANT, API_KEY, DOCS_DIR, JOB_TIMEOUT, AUTO
#
# Pre-requisitos:
#   - MULTIMODAL_ENABLED=true e VISUAL_CATALOG_ENABLED=true, senao a foto nao e
#     aceita e nao ha como identificar o aparelho.
#   - WEB_SEARCH_ENABLED=false no ato 3: com o fallback web ligado, a pergunta
#     sem documento interno pode ser respondida pela busca externa e a
#     demonstracao perde o contraste entre "sem evidencia" e "com evidencia".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/demo-lib.sh
source "$SCRIPT_DIR/demo-lib.sh"

IPHONE_ID="IPHONE-AIR"
MOTO_ID="MOTO-G-2026"
RUN_ID="$(date +%Y%m%d-%H%M%S)"

# As perguntas nao nomeiam o aparelho: quem identifica e a foto anexada.
PERGUNTA_IPHONE="Qual e o peso, a espessura e a protecao de tela deste aparelho?"
PERGUNTA_MOTO="Qual processador e qual versao do Android este aparelho usa?"

# Fotos usadas nas perguntas. A do Moto G e um angulo diferente do cadastrado.
FOTO_IPHONE="iphone.jpeg"
FOTO_MOTO="motog86_2.jpeg"

FOTO_INEDITA=0
ONLY_STEP=""
CLEAN_ONLY=0

for argument in "$@"; do
  case "$argument" in
    --foto-inedita) FOTO_INEDITA=1 ;;
    --limpar) CLEAN_ONLY=1 ;;
    --auto) AUTO=1 ;;
    [0-9]) ONLY_STEP="$argument" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) fail "Argumento desconhecido: $argument" ;;
  esac
done

run_step() { [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "$1" ]; }

limpar() {
  banner "Limpando o acervo da demonstracao"
  info "As mesmas rotas de gestao servem para deixar a base no estado inicial."
  remove_matching documents 'iphone|moto'
  remove_matching images 'iphone|moto'
}

ato_1_carregar_iphone() {
  banner "Ato 1 - Carregando a especificacao e a foto do primeiro aparelho"
  upload_document "$DOCS_DIR/iphone-air-tecnico.md" "$IPHONE_ID"
  upload_image "$DOCS_DIR/$FOTO_IPHONE" "$IPHONE_ID" "Apple iPhone Air" \
    "Apple" "A3517" "Foto de referencia do iPhone Air"
  printf '\n'
  list_indexed documents
  list_indexed images
}

ato_2_perguntar_iphone() {
  banner "Ato 2 - Perguntando com a foto do primeiro aparelho"
  info "A pergunta nao diz qual aparelho e: quem identifica e a imagem."
  ask "demo-$RUN_ID-iphone" "$PERGUNTA_IPHONE" "$DOCS_DIR/$FOTO_IPHONE"
}

ato_3_perguntar_moto_sem_documento() {
  banner "Ato 3 - Mesma pergunta, outra foto, sem manual correspondente"
  info "A foto e de um aparelho cujo manual ainda nao foi carregado."
  info "Sem evidencia interna, a resposta esperada e a recusa segura."
  ask "demo-$RUN_ID-moto-antes" "$PERGUNTA_MOTO" "$DOCS_DIR/$FOTO_MOTO"
}

ato_4_carregar_moto() {
  banner "Ato 4 - Carregando a especificacao e as fotos do segundo aparelho"
  upload_document "$DOCS_DIR/moto-g-2026-tecnico.md" "$MOTO_ID"
  upload_image "$DOCS_DIR/moto.jpeg" "$MOTO_ID" "Motorola Moto G 2026" \
    "Motorola Mobility" "PB9D0020US" "Foto frontal do Moto G 2026"
  if [ "$FOTO_INEDITA" = "1" ]; then
    warn "  --foto-inedita: a foto usada na pergunta nao sera cadastrada."
    info "  O reconhecimento dependera da similaridade com o outro angulo"
    info "  e do limiar VISUAL_MAX_DISTANCE."
  else
    upload_image "$DOCS_DIR/$FOTO_MOTO" "$MOTO_ID" "Motorola Moto G 2026" \
      "Motorola Mobility" "PB9D0020US" "Segundo angulo do Moto G 2026"
  fi
  printf '\n'
  list_indexed documents
  list_indexed images
}

ato_5_perguntar_moto_novamente() {
  banner "Ato 5 - Repetindo a pergunta com a MESMA foto do ato 3"
  info "Nada mudou na pergunta nem na imagem; mudou o acervo indexado."
  ask "demo-$RUN_ID-moto-depois" "$PERGUNTA_MOTO" "$DOCS_DIR/$FOTO_MOTO"
}

main() {
  require_tools
  [ -d "$DOCS_DIR" ] || fail "Pasta de documentos nao encontrada: $DOCS_DIR"

  if [ "$CLEAN_ONLY" = "1" ]; then
    check_api
    limpar
    exit 0
  fi

  check_api

  if [ -z "$ONLY_STEP" ]; then
    limpar
    pause
  fi

  run_step 1 && { ato_1_carregar_iphone; pause; }
  run_step 2 && { ato_2_perguntar_iphone; pause; }
  run_step 3 && { ato_3_perguntar_moto_sem_documento; pause; }
  run_step 4 && { ato_4_carregar_moto; pause; }
  run_step 5 && ato_5_perguntar_moto_novamente

  banner "Fim da demonstracao"
  info "Para repetir do zero: scripts/demo-rag.sh --limpar"
}

main
