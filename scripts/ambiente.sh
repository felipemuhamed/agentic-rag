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
# Testes e demonstracoes (cada um prepara o que precisa e roda sozinho):
#   scripts/ambiente.sh testes        # suite + avaliacao offline + avaliacao visual
#   scripts/ambiente.sh testes --com-demos   # e mais os dois roteiros interativos
#   scripts/ambiente.sh demo          # roteiro multimodal (identifica pela foto)
#   scripts/ambiente.sh monitoracao   # roteiro de alarmes ao vivo, ponta a ponta
#
# Observabilidade e monitoracao:
#   scripts/ambiente.sh langfuse       # sobe a stack self-hosted de tracing
#   scripts/ambiente.sh zabbix-falso   # sobe e configura um Zabbix MCP de MENTIRA
#   scripts/ambiente.sh ferramentas    # lista o que o servidor MCP publica
#
# Opcoes de "subir":
#   --rebuild     reconstroi a imagem sem cache
#   --com-demo    executa scripts/demo-rag.sh assim que a API responder
#
# Opcoes de "monitoracao" e "zabbix-falso":
#   --sem-env     nao altera o .env (so mostra o que faltaria configurar)
#
# Sem argumento, equivale a "subir".

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=scripts/demo-lib.sh
source "$SCRIPT_DIR/demo-lib.sh"

COMPOSE=(docker compose)
READY_TIMEOUT="${READY_TIMEOUT:-600}"

# Endereco do servidor MCP falso dentro da rede do Compose, e a allowlist que a
# demonstracao usa. Deliberadamente sem `host_update`, que o servidor publica.
ZABBIX_FALSO_URL="http://zabbix-falso:8080/mcp"
ZABBIX_FALSO_TOOLS="problem_get,trigger_get,host_get"

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

ENV_PREPARADO=0

preparar_env() {
  # Os comandos se compoem (monitoracao chama zabbix-falso, que chama isto),
  # e repetir os mesmos avisos tres vezes seguidas so treina a pessoa a
  # ignora-los.
  [ "$ENV_PREPARADO" = "1" ] && return 0
  ENV_PREPARADO=1

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

  conferir_monitoracao
  conferir_langfuse

  APP_PORT=$(env_value APP_PORT 8000)
  APP_BIND=$(env_value APP_BIND 127.0.0.1)
  CHROMA_PORT=$(env_value CHROMA_PORT 8001)
  CHROMA_BIND=$(env_value CHROMA_BIND 127.0.0.1)
  API_URL="http://${APP_BIND}:${APP_PORT}"
  CHROMA_URL="http://${CHROMA_BIND}:${CHROMA_PORT}"
}

conferir_monitoracao() {
  local habilitado url ferramentas
  habilitado=$(env_value MONITORING_ENABLED false)
  [ "$habilitado" = "true" ] || return 0

  url=$(env_value MONITORING_MCP_URL)
  ferramentas=$(env_value MONITORING_ALLOWED_TOOLS)
  [ -n "$url" ] \
    || warn "MONITORING_ENABLED=true mas MONITORING_MCP_URL vazio: a monitoracao ficara fora."
  # Uma allowlist vazia nao quebra nada; simplesmente nenhuma ferramenta chega
  # ao modelo, e a falha aparece so na hora da pergunta.
  [ -n "$ferramentas" ] \
    || warn "MONITORING_ALLOWED_TOOLS vazio: nenhuma ferramenta chegara ao modelo (use: scripts/ambiente.sh ferramentas)."
  case "$url" in
    *zabbix-falso*)
      warn "MONITORING_MCP_URL aponta para o Zabbix FALSO da demo (dados inventados)."
      ;;
  esac
}

conferir_langfuse() {
  local habilitado chave segredo
  habilitado=$(env_value LANGFUSE_ENABLED false)
  [ "$habilitado" = "true" ] || return 0

  chave=$(env_value LANGFUSE_PUBLIC_KEY)
  segredo=$(env_value LANGFUSE_SECRET_KEY)
  case "$chave$segredo" in
    ""|*troque_este_valor*)
      warn "LANGFUSE_ENABLED=true com chaves de exemplo: crie um projeto em"
      warn "  $(env_value LANGFUSE_NEXTAUTH_URL http://localhost:3000) e copie as chaves para o .env."
      ;;
  esac
}

# definir_env <CHAVE> <VALOR>
# Escreve a chave no .env, substituindo a linha existente ou acrescentando ao
# fim. Idempotente e anunciado: mexer no .env de alguem sem dizer o que mudou e
# a maneira mais rapida de tornar um ambiente inexplicavel.
definir_env() {
  local chave="$1" valor="$2" atual
  atual=$(env_value "$chave")
  if [ "$atual" = "$valor" ]; then
    info "  $chave ja esta como $valor"
    return 0
  fi
  python3 - "$REPO_ROOT/.env" "$chave" "$valor" <<'PYTHON'
import sys
from pathlib import Path

caminho, chave, valor = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
linhas = caminho.read_text().splitlines()
saida, escrito = [], False
for linha in linhas:
    if linha.split("=", 1)[0].strip() == chave:
        if not escrito:
            saida.append(f"{chave}={valor}")
            escrito = True
        continue
    saida.append(linha)
if not escrito:
    saida.append(f"{chave}={valor}")
caminho.write_text("\n".join(saida) + "\n")
PYTHON
  ok "  $chave=$valor"
}

recriar_api() {
  banner "Recriando a API para ler o .env novo"
  info "Variaveis de ambiente sao lidas na criacao do container: sem recriar,"
  info "a API continua rodando com a configuracao antiga."
  show_cmd "${COMPOSE[*]} up -d --force-recreate api"
  "${COMPOSE[@]}" up -d --force-recreate api || fail "Falha ao recriar a API"
  esperar_api
}

# Verifica se a imagem tem os extras opcionais instalados. O Dockerfile ja
# construiu wheels que ninguem instalava, e o sintoma era a feature sumir em
# silencio, como se o servidor estivesse fora do ar.
conferir_extras() {
  local faltando=""
  if ! "${COMPOSE[@]}" run --rm --entrypoint python api -c \
      "import langfuse, langchain_mcp_adapters" >/dev/null 2>&1; then
    faltando="1"
  fi
  [ -z "$faltando" ] && return 0

  warn "A imagem da API nao tem langfuse/langchain-mcp-adapters: e antiga."
  info "Reconstruindo antes de seguir."
  construir 0
  return 0
}

construir() {
  local rebuild="$1"
  banner "Construindo a imagem da API"
  info "A imagem carrega os extras opcionais (uploads, tracing e o cliente MCP"
  info "de monitoracao), entao mudanca de dependencia exige reconstrucao."
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
  scripts/demo-rag.sh             # roteiro multimodal (identifica pela foto)
  scripts/demo-monitoracao.sh     # roteiro de monitoracao (alarmes ao vivo)
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

comando_langfuse() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env
  banner "Subindo a stack do Langfuse (perfil observability)"
  info "Sao seis containers: web, worker, postgres, clickhouse, minio e redis."
  info "A primeira subida roda migracoes e demora alguns minutos."
  show_cmd "${COMPOSE[*]} --profile observability up -d"
  "${COMPOSE[@]}" --profile observability up -d \
    || fail "Falha ao subir a stack do Langfuse"

  local url
  url=$(env_value LANGFUSE_NEXTAUTH_URL "http://localhost:$(env_value LANGFUSE_PORT 3000)")
  ok "Langfuse subindo em $url"
  if [ "$(env_value LANGFUSE_ENABLED false)" != "true" ]; then
    warn "LANGFUSE_ENABLED=false no .env: a API nao vai enviar tracos."
    info "Crie um projeto na interface, copie as chaves para o .env, ligue a"
    info "flag e recrie a API: ${COMPOSE[*]} up -d --force-recreate api"
  fi
}

# O servidor sobe em poucos segundos, mas listar as ferramentas antes disso
# produz um erro de conexao que parece defeito da integracao.
esperar_zabbix_falso() {
  local url started=$SECONDS codigo
  url="http://$(env_value ZABBIX_FALSO_BIND 127.0.0.1):$(env_value ZABBIX_FALSO_PORT 8080)/mcp"
  printf '%sAguardando o servidor MCP em %s' "$DIM" "$url"
  while :; do
    # Qualquer resposta HTTP serve: o endpoint recusa um GET simples, e a
    # recusa ja prova que ha alguem escutando.
    codigo=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$url" 2>/dev/null)
    if [ -n "$codigo" ] && [ "$codigo" != "000" ]; then
      printf '%s\n' "$RESET"
      ok "  Servidor MCP respondendo em $url"
      return 0
    fi
    if [ $((SECONDS - started)) -ge 60 ]; then
      printf '%s\n' "$RESET"
      "${COMPOSE[@]}" logs --tail 20 zabbix-falso
      fail "Timeout aguardando o zabbix-falso"
    fi
    printf '.'
    sleep 2
  done
}

comando_zabbix_falso() {
  local configurar=1 argumento
  for argumento in "$@"; do
    case "$argumento" in
      --sem-env) configurar=0 ;;
      *) fail "Opcao desconhecida: $argumento" ;;
    esac
  done

  require_tools
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env

  banner "Subindo o Zabbix FALSO da demonstracao"
  warn "Servidor MCP de MENTIRA, com dados inventados. Nunca suba isto ao lado"
  warn "de um ambiente real: serve so para exercitar o caminho da monitoracao"
  warn "numa maquina que nao tem um Zabbix por perto."
  # --force-recreate porque o servidor roda a partir de ./scripts montado: o
  # arquivo muda sem que a configuracao do servico mude, e um `up -d` normal
  # deixaria o container antigo de pe com o codigo velho carregado.
  show_cmd "${COMPOSE[*]} --profile demo up -d --build --force-recreate zabbix-falso"
  "${COMPOSE[@]}" --profile demo up -d --build --force-recreate zabbix-falso \
    || fail "Falha ao subir o zabbix-falso"
  esperar_zabbix_falso

  if [ "$configurar" = "0" ]; then
    warn "--sem-env: o .env nao foi tocado. Para a API usar o servidor falso:"
    info "  MONITORING_ENABLED=true"
    info "  MONITORING_MCP_URL=$ZABBIX_FALSO_URL"
    info "  MONITORING_ALLOWED_TOOLS=$ZABBIX_FALSO_TOOLS"
    return 0
  fi

  banner "Apontando a API para o servidor falso"
  definir_env MONITORING_ENABLED true
  definir_env MONITORING_MCP_URL "$ZABBIX_FALSO_URL"
  definir_env MONITORING_ALLOWED_TOOLS "$ZABBIX_FALSO_TOOLS"
  info "host_update existe no servidor e NAO esta na lista: e a allowlist que"
  info "mantem a escrita fora do alcance de uma pergunta de chat."
}

comando_ferramentas() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env
  banner "Ferramentas publicadas pelo servidor MCP"
  info "A allowlist precisa nomea-las exatamente como o servidor as chama."
  info "Repare no que o servidor publica e a allowlist NAO entrega ao modelo."
  show_cmd "${COMPOSE[*]} run --rm monitoring-tools $*"
  "${COMPOSE[@]}" --profile tools run --rm monitoring-tools "$@"
}

comando_monitoracao() {
  local extras=() argumento
  for argumento in "$@"; do
    case "$argumento" in
      --sem-env) extras+=("$argumento") ;;
      *) fail "Opcao desconhecida: $argumento" ;;
    esac
  done

  require_tools
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env

  # A ordem e a razao de este comando existir: imagem, servidor, .env, API
  # recriada, e so entao a demonstracao. Fora dessa ordem, a API sobe sem as
  # variaveis novas e o roteiro inteiro responde so pela documentacao.
  banner "Preparando a demonstracao de monitoracao"
  garantir_api_de_pe
  conferir_extras
  comando_zabbix_falso "${extras[@]+"${extras[@]}"}"
  recriar_api
  comando_ferramentas
  pause

  AUTO="${AUTO:-0}" API_URL="$API_URL" "$SCRIPT_DIR/demo-monitoracao.sh"
}

comando_demo() {
  require_tools
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env
  garantir_api_de_pe
  AUTO="${AUTO:-0}" API_URL="$API_URL" "$SCRIPT_DIR/demo-rag.sh" "$@"
}

# Sobe a API se ela ainda nao estiver respondendo, em vez de mandar o usuario
# rodar outro comando antes.
garantir_api_de_pe() {
  if curl -sS --max-time 5 "$API_URL/ready" 2>/dev/null | grep -q '"ready"'; then
    ok "API ja respondendo em $API_URL"
    return 0
  fi
  info "API fora do ar: subindo antes de seguir."
  construir 0
  subir_servicos
  esperar_chroma
  esperar_api
}

comando_testes() {
  local com_demos=0 argumento
  for argumento in "$@"; do
    case "$argumento" in
      --com-demos) com_demos=1 ;;
      *) fail "Opcao desconhecida para testes: $argumento" ;;
    esac
  done

  require_tools
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  preparar_env

  testes_unitarios
  avaliacao_offline
  avaliacao_visual

  if [ "$com_demos" = "1" ]; then
    garantir_api_de_pe
    AUTO="${AUTO:-0}" API_URL="$API_URL" "$SCRIPT_DIR/demo-rag.sh"
    comando_monitoracao
  else
    banner "Faltam as duas demonstracoes interativas"
    info "  scripts/ambiente.sh demo         # 5 atos: identificacao pela foto"
    info "  scripts/ambiente.sh monitoracao  # 4 atos: alarmes ao vivo"
    info "Ou rode tudo junto: scripts/ambiente.sh testes --com-demos"
  fi
}

# O interpretador do repositorio, se houver: os testes e as avaliacoes offline
# rodam fora do container, e o venv e onde as dependencias de dev existem.
python_local() {
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    printf '%s' "$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$(command -v python3)"
  fi
}

testes_unitarios() {
  banner "1/3 - Suite automatizada"
  local python_bin
  python_bin=$(python_local)
  if [ -z "$python_bin" ] || ! "$python_bin" -c "import pytest" >/dev/null 2>&1; then
    warn "  pytest nao encontrado. Instale as dependencias de desenvolvimento:"
    show_cmd "python -m venv .venv && .venv/bin/pip install -e '.[dev]'"
    return 0
  fi
  show_cmd "$python_bin -m pytest -q"
  "$python_bin" -m pytest -q || warn "  A suite falhou. Veja a saida acima."
}

avaliacao_offline() {
  banner "2/3 - Avaliacao offline da recuperacao"
  info "Mede as metricas contra o baseline versionado, sem Chroma, sem modelo"
  info "baixado e sem credencial: e o gate que roda em qualquer maquina."
  local python_bin
  python_bin=$(python_local)
  if [ -z "$python_bin" ]; then
    warn "  Nenhum Python encontrado; pulando."
    return 0
  fi
  show_cmd "$python_bin -m app.evaluation.cli evaluation/datasets/project-docs-v1.json --offline --baseline evaluation/baselines/offline-project-docs-v1.json"

  # O relatorio completo tem centenas de linhas de JSON. Na tela fica o que se
  # olha numa apresentacao; o arquivo inteiro segue disponivel para auditoria.
  local relatorio
  relatorio=$(mktemp)
  if ! "$python_bin" -m app.evaluation.cli \
      evaluation/datasets/project-docs-v1.json \
      --offline \
      --name offline-project-docs-v1 \
      --baseline evaluation/baselines/offline-project-docs-v1.json \
      > "$relatorio" 2>&1; then
    warn "  A avaliacao offline falhou:"
    tail -n 20 "$relatorio"
    rm -f "$relatorio"
    return 0
  fi

  "$python_bin" - "$relatorio" <<'PYTHON'
import json
import sys

dados = json.loads(open(sys.argv[1]).read())
resumo = dados["report"]["summary"]
recuperacao = resumo["retrieval"]
respostas = resumo["answers"]
print("  recall@1/3/5 ....... {:.2f} / {:.2f} / {:.2f}".format(
    recuperacao["recall_at_k"]["1"],
    recuperacao["recall_at_k"]["3"],
    recuperacao["recall_at_k"]["5"],
))
print("  MRR ................ {:.2f}".format(recuperacao["mrr"]))
print("  citacoes validas ... {:.2f}".format(respostas["citation_validity"]))

deltas = (dados.get("comparison") or {}).get("metric_deltas") or {}
mudou = [
    f"{grupo}.{nome}={valor:+.3f}"
    for grupo, metricas in deltas.items()
    if isinstance(metricas, dict)
    for nome, valor in metricas.items()
    if isinstance(valor, (int, float)) and abs(valor) > 1e-9 and "latency" not in nome
]
if mudou:
    print("  ATENCAO, mudou contra o baseline: " + ", ".join(mudou))
else:
    print("  sem diferenca de qualidade contra o baseline versionado")
PYTHON
  info "  relatorio completo em $relatorio"
}

avaliacao_visual() {
  banner "3/3 - Avaliacao da identificacao visual"
  info "Varre limiares de VISUAL_MAX_DISTANCE e recomenda um, em vez de"
  info "calibrar no olho. Exige o catalogo visual ja indexado: rode antes"
  info "  scripts/ambiente.sh demo"
  info "para que as fotos de $DOCS_DIR estejam no Chroma."
  show_cmd "${COMPOSE[*]} run --rm evaluate-image /evaluation/datasets/equipment-photos-v1.json --images-root /documents"
  "${COMPOSE[@]}" --profile tools run --rm evaluate-image \
    /evaluation/datasets/equipment-photos-v1.json \
    --images-root /documents \
    --thresholds 0.3 0.4 0.5 0.6 0.7 0.8 \
    || warn "  A avaliacao visual falhou (catalogo vazio?)."
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
  # --remove-orphans derruba tambem o que subiu por perfil (langfuse,
  # zabbix-falso): sem isso eles ficariam de pe apos o "parar".
  show_cmd "${COMPOSE[*]} down --remove-orphans"
  "${COMPOSE[@]}" down --remove-orphans
  ok "Servicos encerrados. Volumes preservados."
}

comando_limpar() {
  detectar_compose
  cd "$REPO_ROOT" || exit 1
  warn "Isto APAGA os volumes: indice do Chroma, checkpoints, arquivos enviados"
  warn "e tambem os dados do Langfuse (tracos, projetos e chaves)."
  if [ "${AUTO:-0}" != "1" ]; then
    printf '%sDigite "apagar" para confirmar:%s ' "$YELLOW" "$RESET"
    local resposta=""
    read -r resposta || true
    [ "$resposta" = "apagar" ] || fail "Cancelado."
  fi
  show_cmd "${COMPOSE[*]} down -v --remove-orphans"
  "${COMPOSE[@]}" down -v --remove-orphans
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
    langfuse|observabilidade) comando_langfuse ;;
    monitoracao|monitoring) comando_monitoracao "$@" ;;
    demo) comando_demo "$@" ;;
    testes|test) comando_testes "$@" ;;
    zabbix-falso|zabbix) comando_zabbix_falso "$@" ;;
    ferramentas|tools) comando_ferramentas "$@" ;;
    status|ps) comando_status ;;
    logs) comando_logs ;;
    parar|down|stop) comando_parar ;;
    limpar|reset) comando_limpar ;;
    -h|--help|ajuda) sed -n '2,30p' "$0" ;;
    *) fail "Comando desconhecido: $comando (use: subir, aquecer, testes, demo, monitoracao, langfuse, zabbix-falso, ferramentas, status, logs, parar, limpar)" ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
