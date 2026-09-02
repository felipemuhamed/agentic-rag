"""Decide whether a question needs live monitoring data, and go get it.

Two separate jobs, deliberately kept apart. The router is a cheap structured
call that answers one yes/no question, so most conversations never pay for tool
loading at all. The investigator only runs after that yes, and is a bounded
tool-calling loop: it collects observations and stops. It never writes the
answer -- that stays with the generator, which is where citation validation and
the refusal path live.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.monitoring.client import MonitoringToolset, MonitoringUnavailable
from app.monitoring.models import MonitoringObservation


logger = logging.getLogger(__name__)

ROUTER_INSTRUCTIONS = """\
Você decide se uma pergunta precisa de dados ao vivo do sistema de monitoração
(alarmes, triggers, eventos, estado de hosts, valores de itens) para ser
respondida.

Responda `needs_monitoring=true` quando a pergunta se referir ao que está
acontecendo agora ou aconteceu recentemente na infraestrutura: alertas, alarmes,
problemas ativos, severidade, hosts afetados, histórico de métricas.

Responda `false` quando a pergunta for sobre procedimento, especificação,
configuração ou qualquer coisa que a documentação responda sem consultar o
estado atual.
"""

INVESTIGATOR_INSTRUCTIONS = """\
Você coleta dados do sistema de monitoração para embasar uma análise. Use as
ferramentas disponíveis para obter exatamente o que a pergunta exige e pare.

Não escreva a análise nem conclusões: outro componente fará isso com os dados que
você coletar. Não invente identificadores; se precisar de um host ou evento
específico, consulte antes. Prefira poucas chamadas bem escolhidas.
"""


class MonitoringDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_monitoring: bool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class StructuredOutputModel(Protocol):
    def with_structured_output(self, schema: type[BaseModel]) -> Any: ...


class MonitoringRouter:
    def __init__(self, model: StructuredOutputModel, *, max_attempts: int = 1):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        structured = model.with_structured_output(MonitoringDecision)
        if hasattr(structured, "with_config"):
            structured = structured.with_config(
                {"tags": ["monitoring-router", "nostream"]}
            )
        if max_attempts > 1 and hasattr(structured, "with_retry"):
            structured = structured.with_retry(stop_after_attempt=max_attempts)
        self._model = structured

    def decide(self, question: str) -> MonitoringDecision:
        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        response = self._model.invoke(
            [
                SystemMessage(content=ROUTER_INSTRUCTIONS),
                HumanMessage(content=f"Pergunta:\n{normalized}"),
            ]
        )
        return MonitoringDecision.model_validate(response)


class ToolBindingModel(Protocol):
    def bind_tools(self, tools: Sequence[Any]) -> Any: ...


class MonitoringInvestigator:
    def __init__(
        self,
        model: ToolBindingModel,
        *,
        toolset: MonitoringToolset,
        max_iterations: int = 3,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        self._model = model
        self._toolset = toolset
        self._max_iterations = max_iterations

    def investigate(self, question: str) -> list[MonitoringObservation]:
        """Collect observations for the question, or none if nothing applies."""
        tools = self._toolset.tools()
        if not tools:
            raise MonitoringUnavailable("no monitoring tools are available")

        bound = self._model.bind_tools(tools)
        if hasattr(bound, "with_config"):
            bound = bound.with_config(
                {"tags": ["monitoring-investigator", "nostream"]}
            )

        messages: list[Any] = [
            SystemMessage(content=INVESTIGATOR_INSTRUCTIONS),
            HumanMessage(content=f"Pergunta do usuário:\n{question.strip()}"),
        ]
        observations: list[MonitoringObservation] = []

        for _ in range(self._max_iterations):
            response = bound.invoke(messages)
            tool_calls = _tool_calls(response)
            if not tool_calls:
                break
            messages.append(response)
            for call in tool_calls:
                observation = self._execute(call)
                observations.append(observation)
                messages.append(
                    ToolMessage(
                        content=observation.content,
                        tool_call_id=str(call.get("id", "")),
                    )
                )

        return observations

    def _execute(self, call: dict[str, Any]) -> MonitoringObservation:
        name = str(call.get("name", ""))
        arguments = call.get("args") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            content = self._toolset.call(name, arguments)
        except Exception as error:
            # A failed call is recorded rather than raised: the model can try a
            # different one, and the generator sees that the datum is missing
            # instead of receiving a confident answer built on nothing. The
            # error status is what later keeps the pipeline from answering an
            # infrastructure question from documentation or the web.
            logger.warning(
                "Monitoring tool failed: %s (%s)",
                name,
                type(error).__name__,
            )
            return MonitoringObservation(
                tool=name,
                arguments=arguments,
                content=json.dumps(
                    {"erro": type(error).__name__, "ferramenta": name},
                    ensure_ascii=False,
                ),
                status="error",
            )
        return MonitoringObservation(
            tool=name,
            arguments=arguments,
            content=content,
        )


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, AIMessage):
        return list(response.tool_calls or [])
    calls = getattr(response, "tool_calls", None)
    return list(calls) if calls else []
