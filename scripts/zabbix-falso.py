#!/usr/bin/env python3
"""Servidor MCP falso que imita um Zabbix, para demonstrar a POC sem um Zabbix.

ISTO NAO E UM ZABBIX. Os dados sao fixos e inventados; serve so para exercitar
o caminho completo -- classificador, allowlist, ponte assincrona, evidencia
citavel -- numa maquina que nao tem um Zabbix real por perto. Em producao,
aponte MONITORING_MCP_URL para o servidor MCP de verdade e esqueca este arquivo.

Duas escolhas sao propositais e importantes para a demonstracao:

* `host_update` existe. Um servidor MCP completo do Zabbix publica a API de
  escrita, e o unico motivo de ela nao chegar ao modelo e a allowlist. Deixar
  a ferramenta aqui e o que torna essa defesa visivel.
* Consultas sem resultado devolvem lista vazia de verdade. E assim que se ve a
  diferenca entre "nao ha alarmes" e "nao consegui perguntar".

Uso:
  python scripts/zabbix-falso.py --host 0.0.0.0 --port 8080
"""

import argparse
import json
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


PROBLEMAS: list[dict[str, Any]] = [
    {
        "eventid": "10501",
        "host": "olt-sp-01",
        "name": "Perda óptica alta na PON 1/1/3",
        "severity": "high",
        "severity_level": 4,
        "clock": "2026-09-01T09:12:44",
        "acknowledged": False,
        "tags": ["optical", "pon"],
    },
    {
        "eventid": "10502",
        "host": "olt-sp-01",
        "name": "Temperatura acima de 65 °C no chassi",
        "severity": "average",
        "severity_level": 3,
        "clock": "2026-09-01T08:47:10",
        "acknowledged": True,
        "tags": ["hardware", "temperature"],
    },
    {
        "eventid": "10503",
        "host": "core-rj-02",
        "name": "Interface Gi0/2 down",
        "severity": "disaster",
        "severity_level": 5,
        "clock": "2026-09-01T09:31:02",
        "acknowledged": False,
        "tags": ["network"],
    },
]

TRIGGERS: list[dict[str, Any]] = [
    {
        "triggerid": "2201",
        "host": "olt-sp-01",
        "description": "Perda óptica alta na PON {#PON}",
        "expression": "last(/olt-sp-01/optical.rx)<-27",
        "priority": 4,
        "status": "enabled",
    },
    {
        "triggerid": "2202",
        "host": "core-rj-02",
        "description": "Interface {#IFNAME} down",
        "expression": "last(/core-rj-02/net.if.status)=2",
        "priority": 5,
        "status": "enabled",
    },
]

HOSTS: list[dict[str, Any]] = [
    {
        "hostid": "301",
        "host": "olt-sp-01",
        "name": "OLT São Paulo 01",
        "equipment_id": "OLT-9000",
        "status": "monitored",
        "groups": ["Acesso/FTTH"],
    },
    {
        "hostid": "302",
        "host": "core-rj-02",
        "name": "Core Rio 02",
        "equipment_id": "CORE-4500",
        "status": "monitored",
        "groups": ["Backbone"],
    },
]

NIVEIS = {
    "not_classified": 0,
    "information": 1,
    "warning": 2,
    "average": 3,
    "high": 4,
    "disaster": 5,
}

# O SDK valida o header Host contra uma allowlist para barrar DNS rebinding, e
# a lista default so aceita localhost. Dentro do Compose o cliente chama
# `http://zabbix-falso:8080/mcp`, entao sem nomear o servico aqui toda chamada
# volta 421 Misdirected Request -- que parece falha da integracao e nao e.
HOSTS_PERMITIDOS = [
    "zabbix-falso",
    "zabbix-falso:*",
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
]

server = FastMCP("zabbix-falso")


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@server.tool()
def problem_get(
    host: str | None = None,
    severity_min: str | None = None,
    limit: int = 10,
) -> str:
    """Lista os problemas ativos, opcionalmente filtrando por host e severidade."""
    minimo = NIVEIS.get((severity_min or "").lower(), 0)
    resultado = [
        problema
        for problema in PROBLEMAS
        if (host is None or problema["host"] == host)
        and problema["severity_level"] >= minimo
    ]
    # Um host desconhecido devolve [] de verdade: e o caso que separa
    # "nao ha alarmes" de "nao consegui consultar".
    return _json(resultado[: max(0, limit)])


@server.tool()
def trigger_get(host: str | None = None, limit: int = 10) -> str:
    """Lista as triggers configuradas, opcionalmente de um host."""
    resultado = [
        trigger for trigger in TRIGGERS if host is None or trigger["host"] == host
    ]
    return _json(resultado[: max(0, limit)])


@server.tool()
def host_get(name: str | None = None) -> str:
    """Lista os hosts monitorados e o equipamento correspondente no acervo."""
    if name is None:
        return _json(HOSTS)
    agulha = name.lower()
    return _json(
        [
            host
            for host in HOSTS
            if agulha in host["host"].lower() or agulha in host["name"].lower()
        ]
    )


@server.tool()
def host_update(hostid: str, status: str) -> str:
    """Altera um host. Existe para provar que a allowlist a mantem fora do chat."""
    return _json({"erro": "servidor de demonstracao: escrita nao executada"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="HOST",
        help=(
            "Nome adicional aceito no header Host (aceita sufixo ':*' para "
            "qualquer porta). Use se o servico tiver outro nome no Compose."
        ),
    )
    argumentos = parser.parse_args()

    permitidos = HOSTS_PERMITIDOS + list(argumentos.allow_host)
    for nome in (argumentos.host, f"{argumentos.host}:*"):
        if nome not in permitidos:
            permitidos.append(nome)

    server.settings.host = argumentos.host
    server.settings.port = argumentos.port
    server.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=permitidos,
        # Nenhum navegador chama este servidor; a origem fica vazia e o
        # validador so reprova um Origin presente e desconhecido.
        allowed_origins=[],
    )
    print(
        f"zabbix-falso em http://{argumentos.host}:{argumentos.port}/mcp "
        "(dados ficticios)",
        flush=True,
    )
    server.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
