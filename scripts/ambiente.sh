#!/usr/bin/env bash
# Sobe, verifica e derruba o ambiente da POC com Docker Compose.
#
# Uso:
#   scripts/ambiente.sh subir     # constroi, sobe chroma + api e espera ficar pronto
#   scripts/ambiente.sh aquecer   # baixa os modelos indexando e removendo um arquivo
#   scripts/ambiente.sh status    # docker compose ps + health/ready
#   scripts/ambiente.sh logs      # acompanha os logs de api e chroma
#   scripts/ambiente.sh parar     # docker compose down (preserva os volumes)
#   scripts/ambiente.sh limpar    # docker compose down -v (APAGA indice e uploads)
#
# Opcoes de "subir":
#   --rebuild     reconstroi a imagem sem cache
#   --com-demo    executa scripts/demo-rag.sh assim que a API responder
#
# Sem argumento, equivale a "subir".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/demo-lib.sh
source "$SCRIPT_DIR/demo-lib.sh"

COMPOSE=(docker compose)
READY_TIMEOUT="${READY_TIMEOUT:-600}"

# Le uma variavel do .env sem executar o arquivo: valores como "Agentic RAG"
# quebrariam um `source`.
env_value() {
  local name="$1" default="${2:-}" value=""
  if [ -f "$REPO_ROOT/.env" ]; then
    value=$(sed -n "s/^[[:space:]]*${name}=//p" "$REPO_ROOT/.env" | tail -n 1)
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
  fi
  printf '%s' "${value:-$default}"
}

detectar_compose() {
  command -v docker >/dev/null 2>&1 \
    || fail "Docker nao encontrado. Instale o Docker Engine para subir o ambiente."
  docker info >/dev/null 2>&1 \
    || fail "O daemon do Docker nao esta respondendo. Inicie o servico e tente de novo."

  if docker compose version >/dev/null 2>&1; then
    COMPOSE=(docker compose)
  elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE=(docker-compose)
  else
    fail "Docker Compose nao encontrado (nem 'docker compose' nem 'docker-compose')."
  fi
  info "Usando: ${COMPOSE[*]} ($("${COMPOSE[@]}" version --short 2>/dev/null || echo '?'))"
}

preparar_env() {
  if [ ! -f "$REPO_ROOT/.env" ]; then
    [ -f "$REPO_ROOT/.env.example" ] \
      || fail "Nem .env nem .env.example existem em $REPO_ROOT"
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    warn ".env criado a partir de .env.example. Preencha LLM_MODEL e a chave do provider."
  fi

  local llm_model provider_key web_search multimodal visual
  llm_model=$(env_value LLM_MODEL)
  provider_key=$(env_value OPENROUTER_API_KEY)
  web_search=$(env_value WEB_SEARCH_ENABLED false)
  multimodal=$(env_value MULTIMODAL_ENABLED true)
  visual=$(env_value VISUAL_CATALOG_ENABLED true)

  [ -n "$llm_model" ] \
    || warn "LLM_MODEL vazio no .env: os uploads funcionam, mas o chat nao respondera."
  [ -n "$provider_key" ] \
    || warn "OPENROUTER_API_KEY vazio no .env: o chat falhara ao chamar o modelo."
  [ "$web_search" = "true" ] \
    && warn "WEB_SEARCH_ENABLED=true: o ato 3 da demo pode ser respondido pela web."
  [ "$multimodal" = "false" ] \
    && warn "MULTIMODAL_ENABLED=false: a demo envia fotos e sera recusada com 422."
  [ "$visual" = "false" ] \
    && warn "VISUAL_CATALOG_ENABLED=false: as fotos nao serao ligadas aos manuais."

  APP_PORT=$(env_value APP_PORT 8000)
  APP_BIND=$(env_value APP_BIND 127.0.0.1)
  CHROMA_PORT=$(env_value CHROMA_PORT 8001)
  CHROMA_BIND=$(env_value CHROMA_BIND 127.0.0.1)
  API_URL="http://${APP_BIND}:${APP_PORT}"
  CHROMA_URL="http://${CHROMA_BIND}:${CHROMA_PORT}"
}

construir() {
  local rebuild="$1"
  banner "Construindo a imagem da API"
  info "As dependencias mudaram (python-multipart e pypdf), entao a imagem"
  info "precisa ser reconstruida antes de subir os endpoints de upload."
  if [ "$rebuild" = "1" ]; then
    show_cmd "${COMPOSE[*]} build --no-cache api"
    "${COMPOSE[@]}" build --no-cache api || fail "Falha ao construir a imagem da API"
  else
    show_cmd "${COMPOSE[*]} build api"
    "${COMPOSE[@]}" build api || fail "Falha ao construir a imagem da API"
  fi
}

subir_servicos() {
  banner "Subindo chroma e api"
  show_cmd "${COMPOSE[*]} up -d api"
  "${COMPOSE[@]}" up -d api || fail "Falha ao subir os servicos"
  printf '\n'
  "${COMPOSE[@]}" ps
}

esperar_chroma() {
  local started=$SECONDS
  printf '%sAguardando o Chroma em %s' "$DIM" "$CHROMA_URL"
  while :; do
    if curl -sS --max-time 3 "$CHROMA_URL/api/v2/healthcheck" >/dev/null 2>&1; then
      printf '%s\n' "$RESET"
      ok "  Chroma respondendo"
      return 0
    fi
    if [ $((SECONDS - started)) -ge 120 ]; then
      printf '%s\n' "$RESET"
      warn "  Chroma ainda nao respondeu na porta publicada; seguindo pela API."
      return 0
    fi
    printf '.'
    sleep 2
  done
}

esperar_api() {
  local started=$SECONDS body
  printf '%sAguardando a API em %s' "$DIM" "$API_URL"
  while :; do
    body=$(curl -sS --max-time 5 "$API_URL/ready" 2>/dev/null)
    if printf '%s' "$body" | grep -q '"ready"'; then
      printf '%s\n' "$RESET"
      ok "  API pronta"
      return 0
    fi
    if ! container_de_pe; then
      printf '%s\n' "$RESET"
      warn "O container da API nao esta de pe. Ultimas linhas do log:"
      "${COMPOSE[@]}" logs --tail 40 api
      fail "A API parou durante a inicializacao"
    fi
    if [ $((SECONDS - started)) -ge "$READY_TIMEOUT" ]; then
      printf '%s\n' "$RESET"
      "${COMPOSE[@]}" logs --tail 40 api
      fail "Timeout aguardando $API_URL/ready"
    fi
    printf '.'
    sleep 3
  done
}

container_de_pe() {
  local estado
  estado=$("${COMPOSE[@]}" ps --format '{{.Service}} {{.State}}' 2>/dev/null \
    | sed -n 's/^api //p' | tail -n 1)
  [ -z "$estado" ] && return 0
  case "$estado" in
    exited*|dead*) return 1 ;;
    *) return 0 ;;
  esac
}

aquecer() {
  banner "Aquecendo os modelos"
  info "O primeiro upload baixa e carrega o modelo de embedding dentro do"
  info "container. Fazendo isso agora, a demonstracao roda em segundos."

  # Arquivos temporarios com nome unico: o aquecimento nunca colide com um
  # documento real do acervo, e a remocao usa o documentId exato do job.
  local trabalho marca documento imagem origem
  trabalho=$(mktemp -d)
  marca="aquecimento-$(date +%Y%m%d%H%M%S)-$$"
  documento="$trabalho/$marca.md"
  printf '# %s\n\nRegistro temporario usado para carregar o modelo de embedding.\n' \
    "$marca" > "$documento"

  JOB_TIMEOUT="${JOB_TIMEOUT:-1800}" upload_document "$documento"
  remove_document documents "$LAST_DOCUMENT_ID"

  origem=$(ls "$DOCS_DIR"/*.jpeg "$DOCS_DIR"/*.jpg "$DOCS_DIR"/*.png 2>/dev/null | head -n 1)
  if [ -n "$origem" ]; then
    imagem="$trabalho/$marca.${origem##*.}"
    cp "$origem" "$imagem"
    JOB_TIMEOUT="${JOB_TIMEOUT:-1800}" upload_image "$imagem" "$marca" \
      "Aquecimento do catalogo visual" "" "" "Registro temporario de aquecimento"
    remove_document images "$LAST_DOCUMENT_ID"
  else
    warn "  Nenhuma imagem em $DOCS_DIR; pulando o aquecimento do modelo visual."
  fi

  rm -rf "$trabalho"
  ok "Modelos carregados e registros temporarios removidos."
}

resumo() {
  banner "Ambiente pronto"
  cat <<RESUMO
  API .......... $API_URL      (health, ready, metrics, chat/stream, documents)
  Chroma ....... $CHROMA_URL
  Documentos ... $REPO_ROOT/$DOCS_DIR

Proximos passos:
  scripts/ambiente.sh aquecer     # opcional, evita esperar o modelo na hora
  scripts/demo-rag.sh             # roteiro da demonstracao
  scripts/ambiente.sh logs        # acompanhar os logs
  scripts/ambiente.sh parar       # encerrar preservando os volumes
RESUMO
}

comando_subir() {
  local rebuild=0 com_demo=0 argumento
  for argumento in "$@"; do
    case "$argumento" in
      --rebuild) rebuild=1 ;;
      --com-demo) com_demo=1 ;;
      *) fail "Opcao desconhecida para subir: $argumento" ;;
    esac
  done

  require_tools
  detectar_compose
  preparar_env
  cd "$REPO_ROOT" || fail "Nao foi possivel entrar em $REPO_ROOT"
  construir "$rebuild"
  subir_servicos
  esperar_chroma
  esperar_api
  resumo

  if [ "$com_demo" = "1" ]; then
    API_URL="$API_URL" "$SCRIPT_DIR/demo-rag.sh"
  fi
}

comando_status() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env
  banner "Servicos"
  "${COMPOSE[@]}" ps
  check_api
}

comando_logs() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  show_cmd "${COMPOSE[*]} logs -f --tail 100 api chroma"
  "${COMPOSE[@]}" logs -f --tail 100 api chroma
}

comando_parar() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  banner "Encerrando o ambiente"
  show_cmd "${COMPOSE[*]} down"
  "${COMPOSE[@]}" down
  ok "Servicos encerrados. Volumes preservados."
}

comando_limpar() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  warn "Isto APAGA os volumes: indice do Chroma, checkpoints e arquivos enviados."
  if [ "${AUTO:-0}" != "1" ]; then
    printf '%sDigite "apagar" para confirmar:%s ' "$YELLOW" "$RESET"
    local resposta=""
    read -r resposta || true
    [ "$resposta" = "apagar" ] || fail "Cancelado."
  fi
  show_cmd "${COMPOSE[*]} down -v"
  "${COMPOSE[@]}" down -v
  ok "Ambiente removido, incluindo os volumes."
}

comando_aquecer() {
  require_tools
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env
  check_api
  aquecer
}

main() {
  local comando="${1:-subir}"
  [ $# -gt 0 ] && shift

  case "$comando" in
    subir|up) comando_subir "$@" ;;
    aquecer|warmup) comando_aquecer ;;
    status|ps) comando_status ;;
    logs) comando_logs ;;
    parar|down|stop) comando_parar ;;
    limpar|reset) comando_limpar ;;
    -h|--help|ajuda) sed -n '2,20p' "$0" ;;
    *) fail "Comando desconhecido: $comando (use: subir, aquecer, status, logs, parar, limpar)" ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
