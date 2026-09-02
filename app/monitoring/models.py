"""Turn a monitoring tool call into evidence the answer can cite.

The pipeline only lets the generator use documents that carry provenance, so
live data has to arrive in the same shape as a manual chunk. Each observation
keeps the tool and the arguments that produced it: an operator reading the
answer can tell that a claim came from `problem_get(limit=10)` and not from the
model's memory, and can reproduce the call.

An observation also records whether the call actually produced data. "Nenhum
alarme" is a fact about the infrastructure and belongs in the answer; "não
consegui perguntar" is the absence of a fact and must never be answered from
another source, because no manual and no page on the web knows the current
state of this customer's infrastructure.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.knowledge.retrieval.models import RetrievedDocument


MONITORING_SOURCE_PREFIX = "zabbix"

ObservationStatus = Literal["ok", "empty", "error"]
MonitoringStatus = Literal["ok", "empty", "unavailable"]

EMPTY_RESULT_NOTE = "Consulta executada. Nenhum resultado retornado."

MONITORING_EMPTY_ANSWER = (
    "Consultei o sistema de monitoração e ele não retornou nenhum dado para "
    "essa pergunta. Como o estado atual da infraestrutura só o monitoramento "
    "conhece, não vou completar a resposta com documentação nem com a web."
)

MONITORING_UNAVAILABLE_ANSWER = (
    "Não consegui consultar o sistema de monitoração agora. Sem esse dado não "
    "tenho como dizer o que está acontecendo na infraestrutura, e responder a "
    "partir de outra fonte seria suposição."
)


def is_empty_payload(content: str) -> bool:
    """Tell an executed-but-empty response from one that carries data.

    Zabbix answers "no problems" with an empty list, which is indistinguishable
    from a real answer unless the emptiness is recognised here.
    """
    normalized = content.strip()
    if not normalized:
        return True
    try:
        payload = json.loads(normalized)
    except (TypeError, ValueError):
        return False
    return payload is None or payload in ([], {}, "")


@dataclass(frozen=True, slots=True)
class MonitoringObservation:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    status: ObservationStatus = "ok"

    def __post_init__(self) -> None:
        # Recognising emptiness here means every construction site -- including
        # tests and future callers -- reports the truth without remembering to.
        if self.status == "ok" and is_empty_payload(self.content):
            object.__setattr__(self, "status", "empty")

    def locator(self) -> str:
        return f"{MONITORING_SOURCE_PREFIX}:{self.tool}"

    def rendered_content(self) -> str:
        """What the generator should read: an empty body says so in words."""
        if self.status == "empty":
            return EMPTY_RESULT_NOTE
        return self.content

    def as_document(self, index: int) -> RetrievedDocument:
        return RetrievedDocument(
            chunk_id=f"{self.locator()}#{index}",
            content=(
                "Dado ao vivo do sistema de monitoração. Trate como observação "
                "do estado atual, não como documentação.\n"
                f"Ferramenta: {self.tool}\n"
                f"Argumentos: {json.dumps(self.arguments, ensure_ascii=False)}\n"
                f"Resposta:\n{self.rendered_content()}"
            ),
            metadata={
                "source": self.locator(),
                "title": f"Monitoração: {self.tool}",
                "section": "Dados ao vivo",
                "media_type": "application/json",
                "tool": self.tool,
            },
            # Live data is the premise of the question, not a similarity hit, so
            # it sits above retrieved chunks without competing with their scores.
            retrieval_score=1.0,
            rerank_score=1.0,
        )


def monitoring_status(
    observations: Sequence[MonitoringObservation],
) -> MonitoringStatus:
    """Summarise what the monitoring pass produced, for the routing decision.

    One useful observation is enough to call the pass successful: a question
    can be answered from the alarms that were found even if another call came
    back empty.
    """
    if not observations:
        return "empty"
    if any(observation.status == "ok" for observation in observations):
        return "ok"
    if all(observation.status == "error" for observation in observations):
        return "unavailable"
    return "empty"


def monitoring_documents(
    observations: Sequence[MonitoringObservation],
) -> list[RetrievedDocument]:
    return [
        observation.as_document(index)
        for index, observation in enumerate(observations)
    ]


def monitoring_context(
    observations: Sequence[MonitoringObservation],
) -> str:
    if not observations:
        return ""
    return json.dumps(
        [
            {
                "ferramenta": observation.tool,
                "argumentos": observation.arguments,
                "resposta": observation.rendered_content(),
            }
            for observation in observations
        ],
        ensure_ascii=False,
        indent=2,
    )
