#!/usr/bin/env bash
# Funcoes compartilhadas pelos scripts de demonstracao da RAG.
# O curl faz todas as chamadas HTTP; python3 e usado apenas para montar e
# formatar JSON, evitando depender de jq na maquina da apresentacao.

# Caminhos resolvidos a partir do proprio arquivo, para que os scripts rodem
# tanto da raiz do repositorio quanto de dentro de scripts/.
DEMO_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_REPO_ROOT="$(cd "$DEMO_LIB_DIR/.." && pwd)"

API_URL="${API_URL:-http://127.0.0.1:8000}"
TENANT="${TENANT:-public}"
API_KEY="${API_KEY:-}"
JOB_TIMEOUT="${JOB_TIMEOUT:-900}"
DOCS_DIR="${DOCS_DIR:-$DEMO_REPO_ROOT/documents}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

AUTH_HEADER=()
if [ -n "$API_KEY" ]; then
  AUTH_HEADER=(-H "X-API-Key: $API_KEY")
fi

banner() { printf '\n%s%s%s\n' "$BOLD$BLUE" "=== $* ===" "$RESET"; }
info()   { printf '%s%s%s\n' "$DIM" "$*" "$RESET"; }
ok()     { printf '%s%s%s\n' "$GREEN" "$*" "$RESET"; }
warn()   { printf '%s%s%s\n' "$YELLOW" "$*" "$RESET"; }
fail()   { printf '%s%s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }
show_cmd() { printf '%s$ %s%s\n' "$DIM" "$*" "$RESET"; }

pause() {
  [ "${AUTO:-0}" = "1" ] && return 0
  printf '\n%s[ENTER para continuar]%s ' "$DIM" "$RESET"
  read -r _ || true
  printf '\n'
}

require_tools() {
  local tool
  for tool in curl python3; do
    command -v "$tool" >/dev/null 2>&1 || fail "Ferramenta obrigatoria ausente: $tool"
  done
}

# Le um campo de um JSON vindo do stdin: echo "$json" | json_get result.chunksWritten
json_get() {
  python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)
for part in sys.argv[1].split("."):
    if data is None:
        break
    if isinstance(data, list):
        try:
            data = data[int(part)]
        except (ValueError, IndexError):
            data = None
    elif isinstance(data, dict):
        data = data.get(part)
    else:
        data = None
print("" if data is None else data)
' "$1"
}

check_api() {
  banner "Verificando a API em $API_URL"
  show_cmd "curl -s $API_URL/health"
  local health ready
  health=$(curl -sS --max-time 10 "$API_URL/health" 2>/dev/null) \
    || fail "API inacessivel em $API_URL. Suba com: docker compose up -d api"
  printf '%s\n' "$health"
  show_cmd "curl -s $API_URL/ready"
  ready=$(curl -sS --max-time 10 "$API_URL/ready" 2>/dev/null) || true
  printf '%s\n' "$ready"
}

# upload_document <arquivo> [equipmentId]
upload_document() {
  local file="$1" equipment="${2:-}"
  [ -f "$file" ] || fail "Arquivo nao encontrado: $file"

  show_cmd "curl -X POST $API_URL/documents -F file=@$file -F tenantId=$TENANT${equipment:+ -F equipmentId=$equipment}"
  local response code body job_id
  response=$(curl -sS -w $'\n%{http_code}' -X POST "$API_URL/documents" \
    "${AUTH_HEADER[@]}" \
    -F "file=@${file}" \
    -F "tenantId=${TENANT}" \
    ${equipment:+-F "equipmentId=${equipment}"})
  code=${response##*$'\n'}
  body=${response%$'\n'*}

  if [ "$code" != "202" ]; then
    printf '%s\n' "$body"
    fail "Upload de $file recusado (HTTP $code)"
  fi
  job_id=$(printf '%s' "$body" | json_get jobId)
  info "202 Accepted -> job $job_id"
  wait_job "$job_id" "$(basename "$file")"
}

# upload_image <arquivo> <equipmentId> <equipmentName> <fabricante> <modelo> <descricao>
upload_image() {
  local file="$1" equipment="$2" name="$3" manufacturer="$4" model="$5" description="$6"
  [ -f "$file" ] || fail "Arquivo nao encontrado: $file"

  show_cmd "curl -X POST $API_URL/documents/images -F file=@$file -F equipmentId=$equipment -F equipmentName='$name'"
  local response code body job_id
  response=$(curl -sS -w $'\n%{http_code}' -X POST "$API_URL/documents/images" \
    "${AUTH_HEADER[@]}" \
    -F "file=@${file}" \
    -F "equipmentId=${equipment}" \
    -F "equipmentName=${name}" \
    -F "manufacturer=${manufacturer}" \
    -F "model=${model}" \
    -F "description=${description}" \
    -F "tenantId=${TENANT}")
  code=${response##*$'\n'}
  body=${response%$'\n'*}

  if [ "$code" != "202" ]; then
    printf '%s\n' "$body"
    fail "Upload da imagem $file recusado (HTTP $code)"
  fi
  job_id=$(printf '%s' "$body" | json_get jobId)
  info "202 Accepted -> job $job_id"
  wait_job "$job_id" "$(basename "$file")"
}

# wait_job <jobId> <rotulo>
wait_job() {
  local job_id="$1" label="${2:-documento}"
  local started=$SECONDS status body chunks

  # Fica disponivel para quem precisa agir sobre o documento recem indexado.
  LAST_DOCUMENT_ID=""

  printf '%sIndexando %s' "$DIM" "$label"
  while :; do
    body=$(curl -sS "${AUTH_HEADER[@]}" \
      "$API_URL/documents/jobs/${job_id}?tenantId=${TENANT}")
    status=$(printf '%s' "$body" | json_get status)

    case "$status" in
      succeeded)
        printf '%s\n' "$RESET"
        chunks=$(printf '%s' "$body" | json_get result.chunksWritten)
        LAST_DOCUMENT_ID=$(printf '%s' "$body" | json_get result.documentId)
        ok "  indexado: $label${chunks:+ ($chunks chunks)}"
        printf '%s' "$body" | python3 -c '
import json
import sys

result = (json.load(sys.stdin).get("result") or {})
print("  documentId:", result.get("documentId"))
if result.get("equipmentId"):
    print("  equipmentId:", result["equipmentId"])
'
        return 0
        ;;
      failed)
        printf '%s\n' "$RESET"
        printf '%s' "$body" | python3 -c '
import json
import sys

job = json.load(sys.stdin)
print("  erro:", job.get("errorCode"), "-", job.get("error"))
'
        fail "Ingestao de $label falhou"
        ;;
      "")
        printf '%s\n' "$RESET"
        printf '%s\n' "$body"
        fail "Nao foi possivel consultar o job $job_id"
        ;;
    esac

    if [ $((SECONDS - started)) -ge "$JOB_TIMEOUT" ]; then
      printf '%s\n' "$RESET"
      fail "Timeout aguardando a ingestao de $label"
    fi
    printf '.'
    sleep 2
  done
}

# ask <conversationId> <pergunta> [arquivo-de-imagem]
ask() {
  local conversation="$1" question="$2" image="${3:-}" payload

  payload=$(python3 -c '
import base64
import json
import sys
from pathlib import Path

conversation, question, tenant = sys.argv[1], sys.argv[2], sys.argv[3]
image = sys.argv[4] if len(sys.argv) > 4 else ""
body = {"conversationId": conversation, "message": question}
if tenant:
    body["tenantId"] = tenant
if image:
    path = Path(image)
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }[path.suffix.lower()]
    body["images"] = [{
        "name": path.name,
        "mimeType": mime,
        "data": base64.b64encode(path.read_bytes()).decode(),
    }]
print(json.dumps(body, ensure_ascii=False))
' "$conversation" "$question" "$TENANT" "$image")

  printf '\n%sPergunta:%s %s\n' "$BOLD" "$RESET" "$question"
  [ -n "$image" ] && info "enviada junto com a imagem $(basename "$image")"
  show_cmd "curl -N -X POST $API_URL/chat/stream -H 'Content-Type: application/json' -d @pergunta.json"

  curl -sS -N -X POST "$API_URL/chat/stream" \
    "${AUTH_HEADER[@]}" \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    --data-binary "$payload" \
  | render_stream
}

# Consome o SSE em um unico processo para que os tokens aparecam em tempo real.
render_stream() {
  python3 -u -c '
import json
import sys

if sys.stdout.isatty():
    BOLD, DIM = "\033[1m", "\033[2m"
    GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
else:
    BOLD = DIM = GREEN = YELLOW = RED = RESET = ""

event = None
sources = []
answer = []
failed = False
raw = []

print(DIM + "--- eventos SSE ---" + RESET)
for line in sys.stdin:
    line = line.rstrip("\n")
    if line.startswith("event: "):
        event = line[7:]
        continue
    if not line.startswith("data: "):
        # Uma resposta que nao e SSE (por exemplo 422 quando a entrada
        # multimodal esta desligada) chega por aqui e precisa ser mostrada.
        if line.strip():
            raw.append(line)
        continue
    data = json.loads(line[6:])

    if event == "status":
        stage = data.get("stage", "")
        message = data.get("message", "")
        print(DIM + "[" + stage + "] " + message + RESET)
    elif event == "sources":
        sources = data.get("sources", [])
    elif event == "token":
        if not answer:
            print("\n" + BOLD + "Resposta:" + RESET, end=" ")
        answer.append(data["text"])
        print(data["text"], end="")
    elif event == "error":
        failed = True
        code = str(data.get("code"))
        message = str(data.get("message"))
        print("\n" + RED + "erro: " + code + " - " + message + RESET)
    elif event == "done":
        print()

if event is None:
    print(RED + "A API nao devolveu um stream SSE:" + RESET)
    for line in raw:
        print("  " + line)
    sys.exit(1)

if sources:
    print("\n" + GREEN + "Fontes citadas: " + str(len(sources)) + RESET)
    for item in sources:
        title = str(item.get("title") or "")
        section = str(item.get("section") or "")
        origin = str(item.get("source") or item.get("url") or "")
        citation = str(item.get("citation") or "")
        suffix = " - " + section if section else ""
        print("  " + citation + " " + title + suffix)
        if origin:
            print(DIM + "      " + origin + RESET)
else:
    print(
        "\n" + YELLOW
        + "Nenhuma fonte: a RAG nao encontrou evidencia interna suficiente."
        + RESET
    )

sys.exit(1 if failed else 0)
'
}

# list_indexed [documents|images]
list_indexed() {
  local kind="${1:-documents}" path="/documents"
  [ "$kind" = "images" ] && path="/documents/images"

  show_cmd "curl -s '$API_URL$path?tenantId=$TENANT'"
  curl -sS "${AUTH_HEADER[@]}" "$API_URL$path?tenantId=$TENANT&limit=100" \
  | python3 -c '
import json
import sys

body = json.load(sys.stdin)
print(str(body.get("total", 0)) + " documento(s) indexado(s):")
for document in body.get("documents", []):
    equipment = document.get("equipmentId") or "-"
    print(
        "  - " + str(document.get("title"))
        + " [" + str(document.get("mediaType")) + "]"
        + " chunks=" + str(document.get("chunkCount"))
        + " equipmentId=" + equipment
    )
    print("      documentId=" + str(document.get("documentId")))
'
}

# remove_document <documents|images> <documentId>
remove_document() {
  local kind="$1" document_id="$2" path="/documents" code
  [ "$kind" = "images" ] && path="/documents/images"
  [ -n "$document_id" ] || return 0

  show_cmd "curl -X DELETE '$API_URL$path/$document_id?tenantId=$TENANT'"
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X DELETE "${AUTH_HEADER[@]}" \
    "$API_URL$path/${document_id}?tenantId=${TENANT}")
  if [ "$code" = "200" ]; then
    info "  removido: $document_id"
  else
    warn "  nao foi possivel remover $document_id (HTTP $code)"
  fi
}

# remove_matching <documents|images> <regex>
remove_matching() {
  local kind="$1" pattern="$2" path="/documents" ids count=0 document_id
  [ "$kind" = "images" ] && path="/documents/images"

  ids=$(curl -sS "${AUTH_HEADER[@]}" "$API_URL$path?tenantId=$TENANT&limit=100" \
    | python3 -c '
import json
import re
import sys

pattern = re.compile(sys.argv[1], re.IGNORECASE)
for document in json.load(sys.stdin).get("documents", []):
    haystack = str(document.get("title", "")) + " " + str(document.get("source", ""))
    if pattern.search(haystack):
        print(document["documentId"])
' "$pattern")

  for document_id in $ids; do
    show_cmd "curl -X DELETE '$API_URL$path/$document_id?tenantId=$TENANT'"
    curl -sS -X DELETE "${AUTH_HEADER[@]}" \
      "$API_URL$path/${document_id}?tenantId=${TENANT}" >/dev/null
    count=$((count + 1))
  done
  info "  removidos de $kind: $count"
}
