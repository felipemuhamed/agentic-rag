#!/usr/bin/env bash
# Demonstracao do roteamento por competencia: a pergunta decide a fonte.
#
#   1. Pergunta de procedimento    -> so a documentacao, o Zabbix nem e tocado
#   2. Pergunta de alarmes         -> consulta o Zabbix e cita `zabbix:` ao vivo
#   3. Pergunta hibrida            -> alarme do Zabbix + manual do equipamento
#   4. Alarme de um host inexistente -> consultou, nao veio nada, e diz isso
#
# O ato 4 e o que separa esta entrega de um chatbot otimista: o Zabbix respondeu
# lista vazia, e a resposta e "nao encontrei", sem reescrever a busca nem cair
# na web. Nenhum manual e nenhuma pagina da internet sabe o estado atual desta
# infraestrutura.
#
# Uso:
#   scripts/demo-monitoracao.sh              # roteiro completo, com pausas
#   AUTO=1 scripts/demo-monitoracao.sh       # sem pausas
#   scripts/demo-monitoracao.sh 2            # executa somente o ato 2
#   scripts/demo-monitoracao.sh --sem-acervo # pula a conferencia do acervo
#
# Variaveis: API_URL, TENANT, API_KEY, DOCS_DIR, JOB_TIMEOUT, AUTO
#
# Pre-requisitos no .env (a checagem abaixo cobra cada um):
#   MONITORING_ENABLED=true
#   MONITORING_MCP_URL=...        # o Zabbix real, ou o falso da POC:
#                                 #   scripts/ambiente.sh zabbix-falso
#   MONITORING_ALLOWED_TOOLS=problem_get,trigger_get,host_get

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/demo-lib.sh
source "$SCRIPT_DIR/demo-lib.sh"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OLT_ID="OLT-9000"

PERGUNTA_DOC="Qual e o procedimento para reiniciar uma OLT com seguranca?"
PERGUNTA_ALARMES="Quais sao os alarmes ativos mais criticos agora?"
PERGUNTA_HIBRIDA="Ha perda optica alta na olt-sp-01? O que devo verificar nesse equipamento?"
PERGUNTA_VAZIA="Existe algum alarme ativo no host olt-rj-99?"

ONLY_STEP=""
SEM_ACERVO=0

for argument in "$@"; do
  case "$argument" in
    --sem-acervo) SEM_ACERVO=1 ;;
    --auto) AUTO=1 ;;
    [0-9]) ONLY_STEP="$argument" ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) fail "Argumento desconhecido: $argument" ;;
  esac
done

run_step() { [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "$1" ]; }

conferir_acervo() {
  local total
  total=$(curl -sS "${AUTH_HEADER[@]}" \
    "$API_URL/documents?tenantId=$TENANT&limit=1" | json_get total)
  if [ "${total:-0}" = "0" ]; then
    warn "  O acervo esta vazio: a resposta sairia so com o alarme."
    info "  Carregue um manual do equipamento antes, por exemplo:"
    show_cmd "curl -X POST $API_URL/documents -F file=@$DOCS_DIR/guia.md -F equipmentId=$OLT_ID -F tenantId=$TENANT"
  else
    info "  $total documento(s) no acervo para complementar o alarme."
  fi
}

verificar_configuracao() {
  banner "Conferindo a configuracao da monitoracao"
  local habilitado url ferramentas
  habilitado=$(env_value MONITORING_ENABLED false)
  url=$(env_value MONITORING_MCP_URL)
  ferramentas=$(env_value MONITORING_ALLOWED_TOOLS)

  [ "$habilitado" = "true" ] || fail \
    "MONITORING_ENABLED=$habilitado no .env. Ligue e recrie a API:
  docker compose up -d --force-recreate api"
  [ -n "$url" ] || fail "MONITORING_MCP_URL vazio no .env."
  if [ -z "$ferramentas" ]; then
    fail "MONITORING_ALLOWED_TOOLS vazio: nenhuma ferramenta chega ao modelo.
  Descubra os nomes reais com:
    scripts/ambiente.sh ferramentas"
  fi

  info "  servidor MCP ..... $url"
  info "  allowlist ........ $ferramentas"
  warn "  Somente estas ferramentas chegam ao modelo, por mais que o servidor"
  warn "  publique. E a allowlist que impede uma pergunta de chat de alcancar"
  warn "  a API de escrita do Zabbix."
}

ato_1_pergunta_de_documentacao() {
  banner "Ato 1 - Pergunta de procedimento"
  info "Nada aqui depende do estado da rede, entao o classificador nao aciona"
  info "a monitoracao: repare que nao aparece o status [monitoring_search]."
  ask "monit-$RUN_ID-doc" "$PERGUNTA_DOC"
}

ato_2_pergunta_de_alarmes() {
  banner "Ato 2 - Pergunta sobre alarmes"
  info "Esta o acervo nao responde por mais completo que seja: a resposta muda"
  info "a cada minuto e esta no Zabbix. Espere [monitoring_search] e fontes"
  info "marcadas como (ao vivo), com o locator zabbix:<ferramenta>."
  ask "monit-$RUN_ID-alarmes" "$PERGUNTA_ALARMES"
}

ato_3_pergunta_hibrida() {
  banner "Ato 3 - Alarme e manual na mesma resposta"
  if [ "$SEM_ACERVO" = "0" ]; then
    # Sem manual indexado a resposta sai so com o alarme, e a demonstracao
    # perde justamente o ponto: as duas fontes na mesma resposta.
    conferir_acervo
  fi
  info "A monitoracao roda ANTES da busca nos manuais: o conteudo do alarme"
  info "entra na consulta que procura a documentacao. Por isso a resposta"
  info "combina o evento ao vivo com o procedimento do manual."
  ask "monit-$RUN_ID-hibrida" "$PERGUNTA_HIBRIDA"
}

ato_4_consulta_sem_resultado() {
  banner "Ato 4 - Consultou o Zabbix e nao veio nada"
  info "O host da pergunta nao existe, entao a consulta roda e devolve vazio."
  warn "Esperado: a resposta diz que nao encontrou, SEM reformular a busca e"
  warn "SEM cair na web. Um alarme ausente nao se responde por analogia."
  ask "monit-$RUN_ID-vazio" "$PERGUNTA_VAZIA"
}

main() {
  require_tools
  check_api
  verificar_configuracao
  [ -z "$ONLY_STEP" ] && pause

  run_step 1 && { ato_1_pergunta_de_documentacao; pause; }
  run_step 2 && { ato_2_pergunta_de_alarmes; pause; }
  run_step 3 && { ato_3_pergunta_hibrida; pause; }
  run_step 4 && ato_4_consulta_sem_resultado

  banner "Fim da demonstracao"
  info "Roteiro multimodal: scripts/demo-rag.sh"
}

main
